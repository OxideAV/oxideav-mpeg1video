//! Per-picture coding parameters threaded through slice / macroblock / block
//! decode so a single code path serves both MPEG-1 and MPEG-2 video.
//!
//! MPEG-1 and MPEG-2 share most of the bitstream (same start codes, same VLC
//! tables for MB type / MBA / motion codes / CBP / DC size / AC run-level)
//! but differ in:
//!   * DC coefficient reset value and decoder-side DC dequantisation
//!     (MPEG-2 uses `intra_dc_precision`: 0..=3 → 8 / 9 / 10 / 11-bit DC).
//!   * AC dequantisation formula (MPEG-2 uses `/32` for non-intra, and drops
//!     the `2 *` factor from the intra formula).
//!   * Mismatch control (MPEG-1 "make odd" per-coefficient; MPEG-2 global
//!     XOR-sum toggle of coeff[63] LSB).
//!   * Escape form for the AC run/level stream (MPEG-1 short/long; MPEG-2
//!     fixed 6-bit run + 12-bit signed level).
//!   * Motion vector f_codes (MPEG-1: one per picture + full_pel flag;
//!     MPEG-2: one per direction per axis, no full_pel).
//!
//! The [`PictureParams`] struct carries the per-picture values, and the
//! [`Codec`] discriminant selects between the two bitstream dialects.

use crate::headers::PictureHeader;

/// Which video-coding dialect this picture belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Mpeg1,
    Mpeg2,
}

/// Direction + axis index helpers for [`PictureParams::f_code`].
pub const DIR_FWD: usize = 0;
pub const DIR_BWD: usize = 1;
pub const AXIS_H: usize = 0;
pub const AXIS_V: usize = 1;

/// H.262 §7.4.2.2 Table 7-6: non-linear `quantiser_scale_code` →
/// `quantiser_scale` map, used when `picture_coding_extension.q_scale_type = 1`.
/// `code = 0` is invalid (slice header forbids it); we map it to 1 to keep
/// downstream arithmetic well-defined and match the existing fallback used
/// by [`PictureParams::quantiser_scale`].
const NON_LINEAR_QUANT_SCALE: [u8; 32] = [
    1, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64,
    72, 80, 88, 96, 104, 112,
];

/// Per-picture coding parameters.
///
/// `f_code[direction][axis]` stores motion-vector f-codes for
/// `direction ∈ {forward, backward}` and `axis ∈ {horizontal, vertical}`.
/// For MPEG-1 pictures both axes carry the picture's single `f_code` and
/// `full_pel_*` is copied from the picture header. For MPEG-2 pictures
/// both `full_pel_*` fields are always `false`.
#[derive(Clone, Copy, Debug)]
pub struct PictureParams {
    pub codec: Codec,
    /// MPEG-2 `intra_dc_precision` (0..=3 → 8/9/10/11-bit DC). Always 0 for
    /// MPEG-1.
    pub intra_dc_precision: u8,
    /// MPEG-2 `alternate_scan` — if set, use the alternate zigzag from
    /// H.262 Figure 7-3 (a.k.a. `scan[1][]`). The decoder honours this flag
    /// per picture; the encoder does not currently emit it.
    pub alternate_scan: bool,
    /// MPEG-2 `intra_vlc_format` — if set, use Table B-15 for intra AC VLC.
    /// First-pass decoder rejects this; first-pass encoder never emits it.
    pub intra_vlc_format: bool,
    /// MPEG-2 `q_scale_type` — if set, use the non-linear quantiser-scale
    /// table from H.262 §7.4.2.2 Table 7-6. Decoder honours this flag per
    /// picture; encoder does not currently emit it.
    pub q_scale_type: bool,
    pub f_code: [[u8; 2]; 2],
    pub full_pel_fwd: bool,
    pub full_pel_bwd: bool,
}

impl PictureParams {
    pub fn is_mpeg2(&self) -> bool {
        self.codec == Codec::Mpeg2
    }

    /// DC predictor reset value at the start of each slice.
    ///
    /// MPEG-1 stores the DC coefficient in pel-space already multiplied by
    /// 8 (a relic of the 8-bit intra_dc_precision), so the reset value is
    /// `128 * 8 = 1024`. MPEG-2 stores the quantised DC directly (`QF[0][0]`)
    /// and the reset value is `128 << intra_dc_precision`.
    pub fn intra_dc_reset_value(&self) -> i32 {
        match self.codec {
            Codec::Mpeg1 => 1024,
            Codec::Mpeg2 => 128 << self.intra_dc_precision,
        }
    }

    /// MPEG-2 `intra_dc_mult` — the multiplier applied to the reconstructed
    /// quantised DC to get the pel-space DC (`intra_dc_mult * QF[0][0]`).
    /// Meaningless (returns 1) for MPEG-1 callers, which pre-scale at the
    /// prediction stage.
    pub fn intra_dc_mult(&self) -> i32 {
        match self.codec {
            Codec::Mpeg1 => 1,
            Codec::Mpeg2 => match self.intra_dc_precision {
                0 => 8,
                1 => 4,
                2 => 2,
                3 => 1,
                _ => 8,
            },
        }
    }

    /// Resolve a 5-bit `quantiser_scale_code` (1..=31) to the actual
    /// `quantiser_scale` for this picture.
    ///
    /// MPEG-1 and MPEG-2 with `q_scale_type = 0` (the linear path) treat the
    /// code as the scale itself; MPEG-2 with `q_scale_type = 1` looks the
    /// code up in H.262 Table 7-6, which lets the scale rise as high as 112
    /// at the cost of resolution at low scales.
    pub fn quantiser_scale(&self, code: u8) -> u8 {
        if self.is_mpeg2() && self.q_scale_type {
            NON_LINEAR_QUANT_SCALE[code as usize]
        } else {
            code
        }
    }

    /// Build MPEG-1 picture parameters from the MPEG-1 picture header.
    pub fn mpeg1_from(ph: &PictureHeader) -> Self {
        let fwd = ph.forward_f_code;
        let bwd = ph.backward_f_code;
        Self {
            codec: Codec::Mpeg1,
            intra_dc_precision: 0,
            alternate_scan: false,
            intra_vlc_format: false,
            q_scale_type: false,
            f_code: [[fwd, fwd], [bwd, bwd]],
            full_pel_fwd: ph.full_pel_forward_vector,
            full_pel_bwd: ph.full_pel_backward_vector,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_reset_values() {
        let mut p = PictureParams {
            codec: Codec::Mpeg1,
            intra_dc_precision: 0,
            alternate_scan: false,
            intra_vlc_format: false,
            q_scale_type: false,
            f_code: [[1, 1], [1, 1]],
            full_pel_fwd: false,
            full_pel_bwd: false,
        };
        assert_eq!(p.intra_dc_reset_value(), 1024);
        p.codec = Codec::Mpeg2;
        p.intra_dc_precision = 0;
        assert_eq!(p.intra_dc_reset_value(), 128);
        p.intra_dc_precision = 1;
        assert_eq!(p.intra_dc_reset_value(), 256);
        p.intra_dc_precision = 2;
        assert_eq!(p.intra_dc_reset_value(), 512);
        p.intra_dc_precision = 3;
        assert_eq!(p.intra_dc_reset_value(), 1024);
    }

    #[test]
    fn quantiser_scale_linear_vs_nonlinear() {
        // MPEG-1 ignores q_scale_type — code is the scale.
        let p1 = PictureParams {
            codec: Codec::Mpeg1,
            intra_dc_precision: 0,
            alternate_scan: false,
            intra_vlc_format: false,
            q_scale_type: false,
            f_code: [[1, 1], [1, 1]],
            full_pel_fwd: false,
            full_pel_bwd: false,
        };
        for code in 1u8..=31 {
            assert_eq!(p1.quantiser_scale(code), code, "MPEG-1 code {code}");
        }

        // MPEG-2 with q_scale_type=0 also linear.
        let p2_lin = PictureParams {
            codec: Codec::Mpeg2,
            ..p1
        };
        for code in 1u8..=31 {
            assert_eq!(p2_lin.quantiser_scale(code), code, "MPEG-2 lin {code}");
        }

        // MPEG-2 with q_scale_type=1 follows H.262 Table 7-6.
        let p2_nl = PictureParams {
            codec: Codec::Mpeg2,
            q_scale_type: true,
            ..p1
        };
        // Spot-check a few entries known to differ from the linear path.
        assert_eq!(p2_nl.quantiser_scale(1), 1);
        assert_eq!(p2_nl.quantiser_scale(8), 8);
        assert_eq!(p2_nl.quantiser_scale(9), 10); // first divergence
        assert_eq!(p2_nl.quantiser_scale(16), 24);
        assert_eq!(p2_nl.quantiser_scale(31), 112); // ceiling
    }

    #[test]
    fn dc_mult_values() {
        let mut p = PictureParams {
            codec: Codec::Mpeg2,
            intra_dc_precision: 0,
            alternate_scan: false,
            intra_vlc_format: false,
            q_scale_type: false,
            f_code: [[1, 1], [1, 1]],
            full_pel_fwd: false,
            full_pel_bwd: false,
        };
        assert_eq!(p.intra_dc_mult(), 8);
        p.intra_dc_precision = 1;
        assert_eq!(p.intra_dc_mult(), 4);
        p.intra_dc_precision = 2;
        assert_eq!(p.intra_dc_mult(), 2);
        p.intra_dc_precision = 3;
        assert_eq!(p.intra_dc_mult(), 1);
    }
}
