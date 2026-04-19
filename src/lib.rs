//! Pure-Rust MPEG-1 (ISO/IEC 11172-2) and MPEG-2 (ISO/IEC 13818-2 / H.262)
//! video decoder and encoder.
//!
//! Current status:
//! * Milestone 1 — sequence / GOP / picture header parsing: done.
//! * Milestone 2 — I-frame decode with intra DCT blocks and YUV 4:2:0 output.
//! * Milestone 3 — P-frames: forward motion compensation (half-pel bilinear),
//!   non-intra block decode, coded_block_pattern handling, skipped-MB forward
//!   propagation.
//! * Milestone 4 — B-frames: forward + backward motion vector decode,
//!   interpolated prediction (averaged), skipped-MB MV inheritance, and
//!   display-order PTS reconstruction from temporal_reference + GOP anchors.
//! * Milestone 5 — I-frame encoder: sequence / GOP / picture headers,
//!   forward DCT, intra quantisation, DC differential + AC run/level VLC,
//!   one-slice-per-row output.
//! * Milestone 6 — P-frame encoder: forward block-matching motion estimation
//!   (integer-pel ±8 + half-pel refinement), MV differential coding via Table
//!   B-10, MB types (skip / forward / forward+coded / intra fallback), inter
//!   quant + Table B-14 first-coeff VLC, and CBP encoding via Table B-9.
//! * Milestone 7 — B-frame encoder: display-order reorder buffer emits
//!   anchors (I/P) before their trailing B-pictures, per-MB decision
//!   between forward / backward / interpolated (fwd+bwd averaged) MC and
//!   intra fallback. Bitstream carries picture_coding_type=3 for B-frames
//!   with forward_f_code and backward_f_code = 1. GOP layout is configurable
//!   via [`encoder::make_encoder_with_gop`] (default: IPP, no B-frames).
//! * Milestone 8 — MPEG-2 video decoder: sequence_extension +
//!   picture_coding_extension parsing, per-direction-per-axis f_codes,
//!   MPEG-2 intra / non-intra dequantisation, MPEG-2 global-XOR mismatch,
//!   MPEG-2 escape (6-bit run + 12-bit signed level). Progressive Main
//!   Profile @ Main Level 4:2:0 only; field pictures, 4:2:2/4:4:4,
//!   intra_vlc_format=1, alternate_scan, non-linear q_scale, dual-prime
//!   and 16×8 MVs are rejected.
//! * Milestone 9 — MPEG-2 I-frame encoder (I-only, progressive 4:2:0).
//!
//! This crate intentionally has no runtime dependencies beyond `oxideav-core`
//! and `oxideav-codec`.

#![allow(clippy::needless_range_loop)]

pub mod block;
pub mod coding_mode;
pub mod dct;
pub mod decoder;
pub mod encoder;
pub mod headers;
pub mod mb;
pub mod motion;
pub mod mpeg2_ext;
pub mod picture;
pub mod start_codes;
pub mod tables;
pub mod vlc;

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecCapabilities, CodecId};

pub const CODEC_ID_STR: &str = "mpeg1video";
pub const CODEC_ID_MPEG2_STR: &str = "mpeg2video";

pub fn register(reg: &mut CodecRegistry) {
    // MPEG-1 video.
    let caps = CodecCapabilities::video("mpeg1video_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(4096, 4096);
    let id = CodecId::new(CODEC_ID_STR);
    reg.register_decoder_impl(id.clone(), caps.clone(), decoder::make_decoder);
    // Encoder produces I + P pictures by default; B-frames are available via
    // `encoder::make_encoder_with_gop`.
    let enc_caps = caps.with_intra_only(false);
    reg.register_encoder_impl(id, enc_caps, encoder::make_encoder);

    // MPEG-2 video (H.262). First-pass milestone: decoder supports I/P/B
    // pictures in progressive 4:2:0 Main Profile; encoder produces I-only
    // bitstreams.
    let caps_m2 = CodecCapabilities::video("mpeg2video_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(1920, 1152);
    let id_m2 = CodecId::new(CODEC_ID_MPEG2_STR);
    reg.register_decoder_impl(id_m2.clone(), caps_m2.clone(), decoder::make_decoder_mpeg2);
    let enc_caps_m2 = caps_m2.with_intra_only(true);
    reg.register_encoder_impl(id_m2, enc_caps_m2, encoder::make_encoder_mpeg2);
}
