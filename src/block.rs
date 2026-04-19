//! Block-level decoding: DC + AC coefficient parsing, dequantisation, IDCT.
//!
//! Shared between MPEG-1 (ISO/IEC 11172-2) and MPEG-2 (H.262 / ISO/IEC
//! 13818-2). The `PictureParams` passed in selects between the two dialects
//! at:
//!   * DC prediction and dequantisation.
//!   * AC dequantisation (scaling divisor, mismatch control).
//!   * Escape coding of run/level pairs.

use oxideav_core::{Error, Result};

use crate::coding_mode::PictureParams;
use crate::dct::idct8x8;
use crate::headers::ZIGZAG;
use crate::tables::dct_coeffs::{self, DctSym};
use crate::tables::dct_dc;
use crate::vlc;
use oxideav_core::bits::BitReader;

/// Sign-extend an `size`-bit unsigned DC-differential value to i32.
fn extend_dc(value: u32, size: u32) -> i32 {
    if size == 0 {
        return 0;
    }
    let vt = 1u32 << (size - 1);
    if value < vt {
        (value as i32) - ((1i32 << size) - 1)
    } else {
        value as i32
    }
}

/// Decode one intra block.
///
/// `prev_dc` carries running DC prediction state for the current component.
/// Its semantics depend on `params.codec`:
///   * MPEG-1: DC is stored in pel-space (already multiplied by 8). Reset
///     value is 1024.
///   * MPEG-2: DC is stored in quantised space (`QF[0][0]`). Reset value is
///     `128 << intra_dc_precision`. The reconstructed pel-space DC is
///     `prev_dc * intra_dc_mult`.
#[allow(clippy::too_many_arguments)]
pub fn decode_intra_block(
    br: &mut BitReader<'_>,
    is_chroma: bool,
    prev_dc: &mut i32,
    quant_scale: u8,
    intra_quant: &[u8; 64],
    params: &PictureParams,
    out_samples: &mut [u8],
    dst_stride: usize,
) -> Result<()> {
    // 1. DC differential.
    let dc_tbl = if is_chroma {
        dct_dc::chroma()
    } else {
        dct_dc::luma()
    };
    let dc_size = vlc::decode(br, dc_tbl)?;
    let dc_diff = if dc_size == 0 {
        0
    } else {
        let bits = br.read_u32(dc_size as u32)?;
        extend_dc(bits, dc_size as u32)
    };

    // Reconstruct the DC coefficient.
    //   * MPEG-1: prev_dc holds pel-space DC (×8 applied). Spec §2.4.4.1:
    //       dct_recon[0][0] = dct_dc_past + dct_dc_differential * 8
    //   * MPEG-2: prev_dc holds QF[0][0] (quantised DC). Spec §7.2.1 /
    //       §7.4.1: QF[0][0] = dc_dct_pred + dct_dc_differential
    //       and F[0][0] = intra_dc_mult * QF[0][0].
    let mut coeffs = [0i32; 64];
    let dc_pel = if params.is_mpeg2() {
        *prev_dc = prev_dc.wrapping_add(dc_diff);
        *prev_dc * params.intra_dc_mult()
    } else {
        let dc_rec = prev_dc.wrapping_add(dc_diff * 8);
        *prev_dc = dc_rec;
        dc_rec
    };
    coeffs[0] = dc_pel;

    // 2. Zig-zag AC coefficients using Table B-14.
    //
    // Per ISO/IEC 11172-2 §2.4.2.9, the AC stream is ALWAYS terminated by an
    // End-Of-Block marker, even when the block holds all 63 AC coefficients.
    // MPEG-2 inherits the same terminator. The first-pass decoder does not
    // support `intra_vlc_format=1` (Table B-15) — it's rejected upstream.
    let ac_tbl = dct_coeffs::table();
    let mut k: usize = 1;
    loop {
        let sym = vlc::decode(br, ac_tbl)?;
        let (run, level) = match sym {
            DctSym::Eob | DctSym::EobOrFirstOne => break,
            DctSym::RunLevel { run, level_abs } => {
                let sign = br.read_u32(1)?;
                let mut lv = level_abs as i32;
                if sign == 1 {
                    lv = -lv;
                }
                (run as usize, lv)
            }
            DctSym::Escape => decode_escape_run_level(br, params)?,
        };
        k += run;
        if k >= 64 {
            return Err(Error::invalid("intra block: AC run past end"));
        }
        // Intra dequantisation:
        //   MPEG-1 §2.4.4.1:  rec = (2 * level * quant * W) / 16
        //   MPEG-2 §7.4.2.3:  rec = (level * quant * W) / 16
        let qf = intra_quant[ZIGZAG[k]] as i32;
        let mut rec = if params.is_mpeg2() {
            (level * quant_scale as i32 * qf) / 16
        } else {
            (2 * level * quant_scale as i32 * qf) / 16
        };
        if !params.is_mpeg2() {
            // MPEG-1 per-coefficient "make odd" mismatch.
            if rec & 1 == 0 && rec != 0 {
                rec = if rec > 0 { rec - 1 } else { rec + 1 };
            }
        }
        rec = rec.clamp(-2048, 2047);
        coeffs[ZIGZAG[k]] = rec;
        k += 1;
    }

    // MPEG-2 §7.4.4 mismatch control: XOR all 64 reconstructed coefficients.
    // If the LSB of the XOR sum is zero, flip the LSB of coeff[63].
    if params.is_mpeg2() {
        apply_mpeg2_mismatch(&mut coeffs);
    }

    // 3. IDCT.
    let mut fblock = [0.0f32; 64];
    for i in 0..64 {
        fblock[i] = coeffs[i] as f32;
    }
    idct8x8(&mut fblock);

    // Write back, clamped to [0,255].
    for j in 0..8 {
        for i in 0..8 {
            let v = fblock[j * 8 + i];
            let px = if v <= 0.0 {
                0
            } else if v >= 255.0 {
                255
            } else {
                v.round() as u8
            };
            out_samples[j * dst_stride + i] = px;
        }
    }
    Ok(())
}

/// Decode one non-intra block. The block does NOT have a DC size/differential
/// prefix — it starts directly with AC coefficients at scan position 0 using
/// the first-coefficient interpretation of Table B-14 (codeword `1s` means
/// `(run=0, level=±1)`, not EOB).
///
/// `prediction` is the motion-compensated prediction samples for this 8×8
/// block; `prediction_stride` gives its row stride. The output is written
/// to `out_samples` as `clamp(prediction + idct(residual), 0, 255)`.
#[allow(clippy::too_many_arguments)]
pub fn decode_non_intra_block(
    br: &mut BitReader<'_>,
    quant_scale: u8,
    non_intra_quant: &[u8; 64],
    params: &PictureParams,
    prediction: &[u8],
    prediction_stride: usize,
    out_samples: &mut [u8],
    dst_stride: usize,
) -> Result<()> {
    let mut coeffs = [0i32; 64];

    // First AC coefficient uses a special table where `1s` decodes to
    // (run=0, level=±1).
    let first_tbl = dct_coeffs::first_coeff_table();
    let ac_tbl = dct_coeffs::table();

    let mut k: usize = 0;
    let mut first = true;
    loop {
        let sym = if first {
            vlc::decode(br, first_tbl)?
        } else {
            vlc::decode(br, ac_tbl)?
        };
        let (run, level) = match sym {
            DctSym::Eob => {
                if first {
                    return Err(Error::invalid("non-intra block: EOB as first symbol"));
                }
                break;
            }
            // Should never fire in these tables — the first-table maps it to
            // RunLevel(0,1) and subsequent uses ac_tbl.
            DctSym::EobOrFirstOne => break,
            DctSym::RunLevel { run, level_abs } => {
                let sign = br.read_u32(1)?;
                let mut lv = level_abs as i32;
                if sign == 1 {
                    lv = -lv;
                }
                (run as usize, lv)
            }
            DctSym::Escape => decode_escape_run_level(br, params)?,
        };
        first = false;
        k += run;
        if k >= 64 {
            return Err(Error::invalid("non-intra block: AC run past end"));
        }
        // Non-intra dequantisation:
        //   MPEG-1 §2.4.4.2:  rec = ((2*level + sign(level)) * quant * W) / 16
        //   MPEG-2 §7.4.2.3:  rec = ((2*level + sign(level)) * quant * W) / 32
        let qf = non_intra_quant[ZIGZAG[k]] as i32;
        let add = if level > 0 { 1 } else { -1 };
        let mut rec = if params.is_mpeg2() {
            ((2 * level + add) * quant_scale as i32 * qf) / 32
        } else {
            ((2 * level + add) * quant_scale as i32 * qf) / 16
        };
        if !params.is_mpeg2() && rec & 1 == 0 && rec != 0 {
            rec = if rec > 0 { rec - 1 } else { rec + 1 };
        }
        rec = rec.clamp(-2048, 2047);
        coeffs[ZIGZAG[k]] = rec;
        k += 1;
    }

    if params.is_mpeg2() {
        apply_mpeg2_mismatch(&mut coeffs);
    }

    // IDCT residual.
    let mut fblock = [0.0f32; 64];
    for i in 0..64 {
        fblock[i] = coeffs[i] as f32;
    }
    idct8x8(&mut fblock);

    // Add prediction and clamp.
    for j in 0..8 {
        for i in 0..8 {
            let p = prediction[j * prediction_stride + i] as i32;
            let r = fblock[j * 8 + i].round() as i32;
            let v = (p + r).clamp(0, 255);
            out_samples[j * dst_stride + i] = v as u8;
        }
    }
    Ok(())
}

/// Copy a prediction block to the output (no residual). Used for non-coded
/// (no-pattern) or skipped macroblocks.
pub fn copy_prediction(
    prediction: &[u8],
    prediction_stride: usize,
    size: usize,
    out_samples: &mut [u8],
    dst_stride: usize,
) {
    for j in 0..size {
        out_samples[j * dst_stride..j * dst_stride + size]
            .copy_from_slice(&prediction[j * prediction_stride..j * prediction_stride + size]);
    }
}

/// Decode an escape run/level pair. The 6-bit escape prefix has already been
/// consumed; this reads run + level and returns them.
///
///   * MPEG-1 §2.4.3.7: `run(6) + 8-bit signed`. `0x00` introduces a long
///     form with an unsigned 8-bit level ∈ 128..=255; `0x80` introduces a
///     long-form negative with 8-bit level ∈ -256..=-129.
///   * MPEG-2 §7.2.2.3: `run(6) + 12-bit signed level`. `0` and `-2048` are
///     forbidden values.
fn decode_escape_run_level(br: &mut BitReader<'_>, params: &PictureParams) -> Result<(usize, i32)> {
    let run = br.read_u32(6)? as usize;
    if params.is_mpeg2() {
        let bits = br.read_u32(12)? as i32;
        // Sign-extend 12-bit two's-complement.
        let level = if bits & 0x800 != 0 {
            bits - 0x1000
        } else {
            bits
        };
        if level == 0 || level == -2048 {
            return Err(Error::invalid("mpeg2 escape: forbidden level (0 or -2048)"));
        }
        Ok((run, level))
    } else {
        let first = br.read_u32(8)? as i32;
        let level = if first == 0 {
            let l = br.read_u32(8)? as i32;
            if l < 128 {
                return Err(Error::invalid("dct escape: long form level < 128"));
            }
            l
        } else if first == 128 {
            let l = br.read_u32(8)? as i32;
            if l > 128 {
                return Err(Error::invalid("dct escape: long form neg level > 128"));
            }
            l - 256
        } else if first >= 128 {
            first - 256
        } else {
            first
        };
        Ok((run, level))
    }
}

/// MPEG-2 mismatch control (§7.4.4): after reconstructing the 8×8 block, XOR
/// every coefficient LSB; if the result is even, flip the LSB of `coeff[63]`.
fn apply_mpeg2_mismatch(coeffs: &mut [i32; 64]) {
    let mut sum: i32 = 0;
    for &c in coeffs.iter() {
        sum ^= c;
    }
    if sum & 1 == 0 {
        // Toggle LSB of last coefficient, keeping saturation in range.
        coeffs[63] ^= 1;
        if coeffs[63] == -2049 {
            coeffs[63] = -2048;
        }
        if coeffs[63] == 2048 {
            coeffs[63] = 2047;
        }
    }
}
