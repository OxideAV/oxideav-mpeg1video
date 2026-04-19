//! MPEG-2 (ISO/IEC 13818-2) extension-header parsing and writing.
//!
//! Extension headers ride on the 0xB5 (`EXTENSION_START_CODE`) marker and
//! carry a 4-bit `extension_start_code_identifier` selecting which extension
//! follows. The first-pass decoder understands:
//!
//! * `sequence_extension` (id = 0x1) — profile, level, progressive, chroma
//!   format, size / VBV / bit-rate extensions, frame-rate extension.
//! * `picture_coding_extension` (id = 0x8) — per-direction-per-axis f_codes,
//!   intra_dc_precision, picture_structure, various per-frame decoding
//!   flags.
//! * `quant_matrix_extension` (id = 0x3) — optional non-default intra and
//!   non-intra quant matrices.
//!
//! All other extension IDs (sequence_display, copyright, picture_display,
//! scalable_*) are parsed only insofar as required to locate the next start
//! code — their payloads are skipped.
//!
//! H.262 §6.2.2.3 / §6.2.3.1 / §6.2.3.2.

use oxideav_core::Result;

use crate::start_codes;
use oxideav_core::bits::BitReader;
use oxideav_core::bits::BitWriter;

/// `sequence_extension()` from H.262 §6.2.2.3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mpeg2SequenceExt {
    pub profile_and_level_indication: u8,
    pub progressive_sequence: bool,
    /// 2-bit `chroma_format`: `01` = 4:2:0, `10` = 4:2:2, `11` = 4:4:4.
    pub chroma_format: u8,
    pub horizontal_size_extension: u8,
    pub vertical_size_extension: u8,
    pub bit_rate_extension: u16,
    pub vbv_buffer_size_extension: u8,
    pub low_delay: bool,
    pub frame_rate_extension_n: u8,
    pub frame_rate_extension_d: u8,
}

impl Default for Mpeg2SequenceExt {
    fn default() -> Self {
        Self {
            // Main Profile @ Main Level: 0x48. The MSB (bit 7) is reserved =
            // `1` for non-escape profile/level; MP@ML is `100_100_0100`
            // which encodes as 0x48.
            profile_and_level_indication: 0x48,
            progressive_sequence: true,
            chroma_format: 0b01,
            horizontal_size_extension: 0,
            vertical_size_extension: 0,
            bit_rate_extension: 0,
            vbv_buffer_size_extension: 0,
            low_delay: false,
            frame_rate_extension_n: 0,
            frame_rate_extension_d: 0,
        }
    }
}

/// `picture_coding_extension()` from H.262 §6.2.3.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mpeg2PictureCodingExt {
    /// `f_code[direction][axis]` — both components are 4-bit values in 1..=15
    /// (or 15 meaning "unused" for I-pictures).
    pub f_code: [[u8; 2]; 2],
    pub intra_dc_precision: u8,
    /// 2-bit `picture_structure`: `01` = top field, `10` = bottom field,
    /// `11` = frame.
    pub picture_structure: u8,
    pub top_field_first: bool,
    pub frame_pred_frame_dct: bool,
    pub concealment_motion_vectors: bool,
    pub q_scale_type: bool,
    pub intra_vlc_format: bool,
    pub alternate_scan: bool,
    pub repeat_first_field: bool,
    pub chroma_420_type: bool,
    pub progressive_frame: bool,
    pub composite_display_flag: bool,
}

impl Default for Mpeg2PictureCodingExt {
    fn default() -> Self {
        Self {
            // Default to "unused" f_codes (15). Encoder overrides per-picture.
            f_code: [[15, 15], [15, 15]],
            intra_dc_precision: 0,
            picture_structure: 0b11,
            top_field_first: true,
            frame_pred_frame_dct: true,
            concealment_motion_vectors: false,
            q_scale_type: false,
            intra_vlc_format: false,
            alternate_scan: false,
            repeat_first_field: false,
            chroma_420_type: true,
            progressive_frame: true,
            composite_display_flag: false,
        }
    }
}

/// Non-default quantiser matrices carried by the `quant_matrix_extension`.
#[derive(Clone, Debug, Default)]
pub struct Mpeg2QuantMatrixExt {
    /// `Some(m)` when `load_intra_quantiser_matrix` = 1.
    pub intra: Option<[u8; 64]>,
    /// `Some(m)` when `load_non_intra_quantiser_matrix` = 1.
    pub non_intra: Option<[u8; 64]>,
    // Chroma matrices (4:2:2 / 4:4:4) are parsed but not retained in the
    // first-pass subset.
}

/// Outcome of parsing a single extension payload.
pub enum ParsedExt {
    Sequence(Mpeg2SequenceExt),
    PictureCoding(Mpeg2PictureCodingExt),
    QuantMatrix(Mpeg2QuantMatrixExt),
    /// Known extension whose body was consumed but whose content is not
    /// retained by the first-pass decoder.
    Other(u8),
}

/// Parse one extension payload. `br` is positioned just after the 4-byte
/// `0x000001B5` start code; the first 4 bits read are
/// `extension_start_code_identifier`.
pub fn parse_extension(br: &mut BitReader<'_>) -> Result<ParsedExt> {
    let id = br.read_u32(4)? as u8;
    match id {
        start_codes::EXT_ID_SEQUENCE => {
            let profile_and_level_indication = br.read_u32(8)? as u8;
            let progressive_sequence = br.read_u32(1)? == 1;
            let chroma_format = br.read_u32(2)? as u8;
            let horizontal_size_extension = br.read_u32(2)? as u8;
            let vertical_size_extension = br.read_u32(2)? as u8;
            let bit_rate_extension = br.read_u32(12)? as u16;
            let _marker = br.read_u32(1)?;
            let vbv_buffer_size_extension = br.read_u32(8)? as u8;
            let low_delay = br.read_u32(1)? == 1;
            let frame_rate_extension_n = br.read_u32(2)? as u8;
            let frame_rate_extension_d = br.read_u32(5)? as u8;
            Ok(ParsedExt::Sequence(Mpeg2SequenceExt {
                profile_and_level_indication,
                progressive_sequence,
                chroma_format,
                horizontal_size_extension,
                vertical_size_extension,
                bit_rate_extension,
                vbv_buffer_size_extension,
                low_delay,
                frame_rate_extension_n,
                frame_rate_extension_d,
            }))
        }
        start_codes::EXT_ID_PICTURE_CODING => {
            let f00 = br.read_u32(4)? as u8;
            let f01 = br.read_u32(4)? as u8;
            let f10 = br.read_u32(4)? as u8;
            let f11 = br.read_u32(4)? as u8;
            let intra_dc_precision = br.read_u32(2)? as u8;
            let picture_structure = br.read_u32(2)? as u8;
            let top_field_first = br.read_u32(1)? == 1;
            let frame_pred_frame_dct = br.read_u32(1)? == 1;
            let concealment_motion_vectors = br.read_u32(1)? == 1;
            let q_scale_type = br.read_u32(1)? == 1;
            let intra_vlc_format = br.read_u32(1)? == 1;
            let alternate_scan = br.read_u32(1)? == 1;
            let repeat_first_field = br.read_u32(1)? == 1;
            let chroma_420_type = br.read_u32(1)? == 1;
            let progressive_frame = br.read_u32(1)? == 1;
            let composite_display_flag = br.read_u32(1)? == 1;
            if composite_display_flag {
                // v_axis(1), field_sequence(3), sub_carrier(1), burst_amplitude(7), sub_carrier_phase(8) = 20 bits
                br.skip(20)?;
            }
            Ok(ParsedExt::PictureCoding(Mpeg2PictureCodingExt {
                f_code: [[f00, f01], [f10, f11]],
                intra_dc_precision,
                picture_structure,
                top_field_first,
                frame_pred_frame_dct,
                concealment_motion_vectors,
                q_scale_type,
                intra_vlc_format,
                alternate_scan,
                repeat_first_field,
                chroma_420_type,
                progressive_frame,
                composite_display_flag,
            }))
        }
        start_codes::EXT_ID_QUANT_MATRIX => {
            let load_intra = br.read_u32(1)? == 1;
            let mut intra = None;
            if load_intra {
                let mut m = [0u8; 64];
                for slot in m.iter_mut().take(64) {
                    *slot = br.read_u32(8)? as u8;
                }
                intra = Some(m);
            }
            let load_non_intra = br.read_u32(1)? == 1;
            let mut non_intra = None;
            if load_non_intra {
                let mut m = [0u8; 64];
                for slot in m.iter_mut().take(64) {
                    *slot = br.read_u32(8)? as u8;
                }
                non_intra = Some(m);
            }
            // `load_chroma_intra_quantiser_matrix` and
            // `load_chroma_non_intra_quantiser_matrix` exist only for
            // 4:2:2 / 4:4:4. In the 4:2:0 first-pass subset we don't
            // reach this extension with those bits set.
            let load_chroma_intra = br.read_u32(1).unwrap_or(0) == 1;
            if load_chroma_intra {
                for _ in 0..64 {
                    br.read_u32(8)?;
                }
            }
            let load_chroma_non_intra = br.read_u32(1).unwrap_or(0) == 1;
            if load_chroma_non_intra {
                for _ in 0..64 {
                    br.read_u32(8)?;
                }
            }
            Ok(ParsedExt::QuantMatrix(Mpeg2QuantMatrixExt {
                intra,
                non_intra,
            }))
        }
        other => {
            // Not parsed — caller skips to next start code by scanning bytes.
            Ok(ParsedExt::Other(other))
        }
    }
}

/// Emit a `sequence_extension()` payload (without the leading start code).
/// Caller is responsible for writing `0x000001B5` first.
pub fn write_sequence_extension(bw: &mut BitWriter, ext: &Mpeg2SequenceExt) {
    bw.write_bits(start_codes::EXT_ID_SEQUENCE as u32, 4);
    bw.write_bits(ext.profile_and_level_indication as u32, 8);
    bw.write_bits(ext.progressive_sequence as u32, 1);
    bw.write_bits(ext.chroma_format as u32, 2);
    bw.write_bits(ext.horizontal_size_extension as u32, 2);
    bw.write_bits(ext.vertical_size_extension as u32, 2);
    bw.write_bits(ext.bit_rate_extension as u32, 12);
    bw.write_bits(1, 1); // marker
    bw.write_bits(ext.vbv_buffer_size_extension as u32, 8);
    bw.write_bits(ext.low_delay as u32, 1);
    bw.write_bits(ext.frame_rate_extension_n as u32, 2);
    bw.write_bits(ext.frame_rate_extension_d as u32, 5);
}

/// Emit a `picture_coding_extension()` payload (without the leading start
/// code). Caller is responsible for writing `0x000001B5` first.
pub fn write_picture_coding_extension(bw: &mut BitWriter, ext: &Mpeg2PictureCodingExt) {
    bw.write_bits(start_codes::EXT_ID_PICTURE_CODING as u32, 4);
    bw.write_bits(ext.f_code[0][0] as u32, 4);
    bw.write_bits(ext.f_code[0][1] as u32, 4);
    bw.write_bits(ext.f_code[1][0] as u32, 4);
    bw.write_bits(ext.f_code[1][1] as u32, 4);
    bw.write_bits(ext.intra_dc_precision as u32, 2);
    bw.write_bits(ext.picture_structure as u32, 2);
    bw.write_bits(ext.top_field_first as u32, 1);
    bw.write_bits(ext.frame_pred_frame_dct as u32, 1);
    bw.write_bits(ext.concealment_motion_vectors as u32, 1);
    bw.write_bits(ext.q_scale_type as u32, 1);
    bw.write_bits(ext.intra_vlc_format as u32, 1);
    bw.write_bits(ext.alternate_scan as u32, 1);
    bw.write_bits(ext.repeat_first_field as u32, 1);
    bw.write_bits(ext.chroma_420_type as u32, 1);
    bw.write_bits(ext.progressive_frame as u32, 1);
    bw.write_bits(ext.composite_display_flag as u32, 1);
    // Never emit composite-display payload from this encoder.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_sequence_extension() {
        let ext = Mpeg2SequenceExt::default();
        let mut bw = BitWriter::new();
        // Pretend we're already byte-aligned (no prior payload).
        write_sequence_extension(&mut bw, &ext);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        match parse_extension(&mut br).unwrap() {
            ParsedExt::Sequence(parsed) => assert_eq!(parsed, ext),
            _ => panic!("wrong extension id"),
        }
    }

    #[test]
    fn round_trip_picture_coding_extension_iframe() {
        let ext = Mpeg2PictureCodingExt {
            f_code: [[15, 15], [15, 15]],
            intra_dc_precision: 0,
            picture_structure: 0b11,
            top_field_first: true,
            frame_pred_frame_dct: true,
            concealment_motion_vectors: false,
            q_scale_type: false,
            intra_vlc_format: false,
            alternate_scan: false,
            repeat_first_field: false,
            chroma_420_type: true,
            progressive_frame: true,
            composite_display_flag: false,
        };
        let mut bw = BitWriter::new();
        write_picture_coding_extension(&mut bw, &ext);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        match parse_extension(&mut br).unwrap() {
            ParsedExt::PictureCoding(parsed) => assert_eq!(parsed, ext),
            _ => panic!("wrong extension id"),
        }
    }
}
