//! MPEG-1 video encoder (ISO/IEC 11172-2) — I + P + B pictures.
//!
//! Scope:
//! * Sequence header (resolution, frame rate, aspect ratio, bit rate, VBV).
//! * GOP header (closed GOP, time-code 0).
//! * Per-picture coding type 1 (I), 2 (P) or 3 (B).
//! * One slice per macroblock row.
//! * Intra macroblocks only for I-pictures.
//! * For P-pictures, four MB types:
//!     * Skipped (MBA increment, MV=(0,0), no residual).
//!     * `MB_INTRA` (fall-back to intra coding).
//!     * `MB_FORWARD` (forward MC, no coded residual; CBP=0, MB type "001").
//!     * `MB_FORWARD + PATTERN` (forward MC + coded residual).
//! * For B-pictures, bidirectional motion-compensated prediction per MB:
//!     * Forward (fwd-only), Backward (bwd-only), Interpolated (average of
//!       fwd + bwd) and INTRA fallback. Each of the inter options may carry
//!       a coded residual (CBP via Table B-9, non-intra quant + Table B-14
//!       VLC).
//! * GOP reorder buffer: the encoder accepts frames in display order and
//!   emits them in bitstream order (anchor first, then its preceding Bs).
//! * Block-matching motion estimation at integer-pel ±8 with half-pel
//!   refinement. With f_code=1 the motion-vector code range is ±16 half-pel.
//! * MV differential encoding via Table B-10 + sign bit (forward_f_code =
//!   backward_f_code = 1 → no complement_r bits).
//! * Inter-block residual: forward DCT of (sample - prediction), then
//!   non-intra quantisation, then run/level VLC via Table B-14 with the
//!   "first coefficient" interpretation (1s = ±1 instead of EOB).
//! * 4:2:0 chroma subsampling.
//!
//! The encoder maintains *reconstructed* reference pictures (forward and
//! backward slots for B-frame encoding) so that the prediction it builds is
//! bit-exact w.r.t. what the decoder will see — this is essential for
//! drift-free round-trips.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Rational, Result,
    TimeBase, VideoFrame,
};

use crate::coding_mode::Codec;
use crate::dct::{fdct8x8, idct8x8};
use crate::headers::{DEFAULT_INTRA_QUANT, DEFAULT_NON_INTRA_QUANT, ZIGZAG};
use crate::mpeg2_ext::{
    write_picture_coding_extension, write_sequence_extension, Mpeg2PictureCodingExt,
    Mpeg2SequenceExt,
};
use crate::start_codes::{
    EXTENSION_START_CODE, GROUP_START_CODE, PICTURE_START_CODE, SEQUENCE_END_CODE,
    SEQUENCE_HEADER_CODE,
};
use crate::tables::dct_coeffs::{self, DctSym};
use crate::tables::dct_dc;
use crate::tables::mba;
use crate::tables::motion as mv_tbl;
use crate::tables::{cbp as cbp_tbl, mb_type};
use crate::vlc::VlcEntry;
use oxideav_core::bits::BitWriter;

/// Default fixed quantiser scale. The lower this is, the finer the
/// quantisation step (less coding loss, more bits per frame).
pub const DEFAULT_QUANT_SCALE: u8 = 3;

/// Default GOP size (number of pictures per GOP). The first picture of each
/// GOP is an I-frame; the remainder are P/B frames laid out per the
/// [`DEFAULT_NUM_B_FRAMES`] pattern.
///
/// The default is intentionally short to keep cumulative drift in the f32
/// IDCT chain bounded. Production users should set a larger GOP via
/// [`make_encoder_with_gop`].
pub const DEFAULT_GOP_SIZE: u32 = 3;

/// Default number of B-frames between two consecutive anchor (I/P) frames.
/// `0` gives the classic `IPPP...` GOP layout (no B-frames). To enable a
/// B-frame GOP (e.g. `IBBP` with `num_b_frames = 2`) use
/// [`make_encoder_with_gop`].
pub const DEFAULT_NUM_B_FRAMES: u32 = 0;

/// Maximum |motion_code| after differential — Table B-10 has entries
/// 0..=16, so 16 is the spec limit for f_code=1.
const MAX_MOTION_CODE: i32 = 16;

/// Encoder factory used by `register()`.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    make_encoder_with_gop(params, DEFAULT_GOP_SIZE, DEFAULT_NUM_B_FRAMES)
}

/// MPEG-2 encoder factory used by `register()`. First-pass milestone: I-only
/// (B-frames disabled, GOP = 1), progressive, 4:2:0, Main Profile @ Main
/// Level. `intra_vlc_format` / `alternate_scan` / non-linear `q_scale_type`
/// are never emitted.
pub fn make_encoder_mpeg2(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    make_encoder_mpeg2_with_gop(params, 1, 0)
}

/// MPEG-2 encoder factory that allows GOP customisation. Currently rejects
/// `num_b_frames > 0` — P/B encoding for MPEG-2 is a later milestone.
pub fn make_encoder_mpeg2_with_gop(
    params: &CodecParameters,
    gop_size: u32,
    num_b_frames: u32,
) -> Result<Box<dyn Encoder>> {
    if num_b_frames != 0 {
        return Err(Error::unsupported(
            "MPEG-2 encoder: B-frames not supported in this milestone",
        ));
    }
    if gop_size != 1 {
        return Err(Error::unsupported(
            "MPEG-2 encoder: only I-only GOPs (gop_size = 1) supported in this milestone",
        ));
    }
    let enc = build_encoder(params, gop_size, num_b_frames, Codec::Mpeg2)?;
    Ok(Box::new(enc))
}

/// Encoder factory allowing callers to override the GOP size and B-frame
/// spacing. `num_b_frames` is the number of B-frames between two
/// consecutive anchor (I/P) frames: `IBBP` corresponds to `num_b_frames = 2`.
///
/// `num_b_frames = 0` disables B-frames entirely (classic `IPPP` GOP).
pub fn make_encoder_with_gop(
    params: &CodecParameters,
    gop_size: u32,
    num_b_frames: u32,
) -> Result<Box<dyn Encoder>> {
    let enc = build_encoder(params, gop_size, num_b_frames, Codec::Mpeg1)?;
    Ok(Box::new(enc))
}

fn build_encoder(
    params: &CodecParameters,
    gop_size: u32,
    num_b_frames: u32,
    codec: Codec,
) -> Result<Mpeg1VideoEncoder> {
    let label = match codec {
        Codec::Mpeg1 => "MPEG-1",
        Codec::Mpeg2 => "MPEG-2",
    };
    let width = params
        .width
        .ok_or_else(|| Error::invalid(format!("{label} encoder: missing width")))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid(format!("{label} encoder: missing height")))?;
    if width == 0 || height == 0 {
        return Err(Error::invalid(format!("{label} encoder: zero-sized frame")));
    }
    if width > 4095 || height > 4095 {
        return Err(Error::invalid(format!(
            "{label} encoder: dimensions exceed 12-bit"
        )));
    }
    if gop_size == 0 {
        return Err(Error::invalid(format!(
            "{label} encoder: gop_size must be ≥ 1"
        )));
    }
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    if pix != PixelFormat::Yuv420P {
        return Err(Error::unsupported(format!(
            "{label} encoder: only Yuv420P supported (got {:?})",
            pix
        )));
    }
    let frame_rate = params.frame_rate.unwrap_or(Rational::new(25, 1));
    let frame_rate_code = frame_rate_code_for(frame_rate)
        .ok_or_else(|| Error::invalid(format!("{label} encoder: unsupported frame rate")))?;
    let bit_rate = params.bit_rate.unwrap_or(1_500_000);

    let codec_id_str = match codec {
        Codec::Mpeg1 => super::CODEC_ID_STR,
        Codec::Mpeg2 => super::CODEC_ID_MPEG2_STR,
    };
    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(codec_id_str);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(PixelFormat::Yuv420P);
    output_params.frame_rate = Some(frame_rate);
    output_params.bit_rate = Some(bit_rate);

    let time_base = TimeBase::new(frame_rate.den, frame_rate.num);

    Ok(Mpeg1VideoEncoder {
        codec,
        output_params,
        width,
        height,
        frame_rate_code,
        bit_rate,
        quant_scale: DEFAULT_QUANT_SCALE,
        gop_size,
        num_b_frames,
        time_base,
        pending: VecDeque::new(),
        gop_pos: 0,
        ref_y: Vec::new(),
        ref_cb: Vec::new(),
        ref_cr: Vec::new(),
        ref_y_stride: 0,
        ref_c_stride: 0,
        ref_valid: false,
        prev_ref_y: Vec::new(),
        prev_ref_cb: Vec::new(),
        prev_ref_cr: Vec::new(),
        prev_ref_y_stride: 0,
        prev_ref_c_stride: 0,
        prev_ref_valid: false,
        b_queue: VecDeque::new(),
        eof: false,
        finalised: false,
    })
}

/// Map an `(num, den)` frame rate to MPEG-1 `frame_rate_code` (Table 2-D.4).
fn frame_rate_code_for(r: Rational) -> Option<u8> {
    let approx = r.num as f64 / r.den as f64;
    let pairs: &[(u8, f64)] = &[
        (1, 24000.0 / 1001.0),
        (2, 24.0),
        (3, 25.0),
        (4, 30000.0 / 1001.0),
        (5, 30.0),
        (6, 50.0),
        (7, 60000.0 / 1001.0),
        (8, 60.0),
    ];
    for (code, fr) in pairs {
        if (approx - fr).abs() < 0.001 {
            return Some(*code);
        }
    }
    None
}

/// A buffered input frame with its intended display-order temporal reference.
struct QueuedFrame {
    frame: VideoFrame,
    /// `temporal_reference` = display-order position within the current GOP.
    temporal_reference: u16,
}

struct Mpeg1VideoEncoder {
    /// MPEG-1 or MPEG-2 bitstream. For MPEG-2 the encoder emits
    /// sequence_extension + picture_coding_extension, uses the MPEG-2 intra
    /// dequant and mismatch formulas, and writes escape run/level pairs in
    /// the MPEG-2 12-bit form.
    codec: Codec,
    output_params: CodecParameters,
    width: u32,
    height: u32,
    frame_rate_code: u8,
    bit_rate: u64,
    quant_scale: u8,
    /// Pictures per GOP in display order. Must be ≥ 1.
    gop_size: u32,
    /// Number of B-frames between two consecutive anchor (I/P) frames.
    /// Anchor distance `m = num_b_frames + 1`.
    num_b_frames: u32,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    /// Position within the current GOP in *display* order. Picture 0 is I.
    gop_pos: u32,
    /// Reconstructed "backward" reference picture = most recently encoded
    /// anchor (I or P). Used as the forward reference for the *next* P
    /// frame, and as the backward reference for any buffered B frames.
    /// Plane sizes are macroblock-aligned.
    ref_y: Vec<u8>,
    ref_cb: Vec<u8>,
    ref_cr: Vec<u8>,
    ref_y_stride: usize,
    ref_c_stride: usize,
    /// True once we have at least one I-picture in the `ref_*` slot.
    ref_valid: bool,
    /// Reconstructed "forward" reference picture = penultimate anchor. Used
    /// as the forward reference for any buffered B frames.
    prev_ref_y: Vec<u8>,
    prev_ref_cb: Vec<u8>,
    prev_ref_cr: Vec<u8>,
    prev_ref_y_stride: usize,
    prev_ref_c_stride: usize,
    prev_ref_valid: bool,
    /// Display-order B-frame reorder buffer. B-frames arrive from the
    /// caller before their backward reference exists; we hold them here
    /// until the next anchor has been encoded, then flush them.
    b_queue: VecDeque<QueuedFrame>,
    eof: bool,
    finalised: bool,
}

impl Encoder for Mpeg1VideoEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::invalid("MPEG-1 encoder: video frames only")),
        };
        if v.width != self.width || v.height != self.height {
            return Err(Error::invalid(
                "MPEG-1 encoder: frame dimensions do not match encoder config",
            ));
        }
        if v.format != PixelFormat::Yuv420P {
            return Err(Error::invalid(
                "MPEG-1 encoder: only Yuv420P input frames supported",
            ));
        }
        if v.planes.len() != 3 {
            return Err(Error::invalid("MPEG-1 encoder: expected 3 planes"));
        }

        let pos = self.gop_pos;
        let tr = pos as u16;
        let kind = picture_kind_for_position(pos, self.num_b_frames, self.ref_valid);

        match kind {
            PictureKind::I | PictureKind::P => {
                let is_intra = matches!(kind, PictureKind::I);
                // If this is the start of a new GOP (I-frame at pos 0) and
                // there are pending B-frames from the previous GOP, promote
                // them to P-frames before emitting the new I. Rationale:
                // B-frames that straddle GOP boundaries would need the new I
                // as their backward reference, but their temporal_reference
                // belongs to the previous GOP. Promoting them to P keeps the
                // GOP structure "closed" and avoids PTS-reconstruction
                // ambiguity on the decoder side.
                if is_intra && pos == 0 && !self.b_queue.is_empty() {
                    self.finalise_trailing_b_as_p()?;
                }
                let data = encode_anchor_picture(self, v, is_intra, tr)?;
                let mut pkt = Packet::new(0, self.time_base, data);
                pkt.pts = v.pts;
                pkt.dts = v.pts;
                pkt.flags.keyframe = is_intra;
                self.pending.push_back(pkt);
                // Flush any buffered B-frames that are display-ordered BEFORE
                // this anchor. They use prev_ref as fwd and the freshly-rolled
                // ref as bwd.
                self.flush_buffered_b_frames()?;
            }
            PictureKind::B => {
                // Buffer until the next anchor has been emitted.
                self.b_queue.push_back(QueuedFrame {
                    frame: v.clone(),
                    temporal_reference: tr,
                });
            }
        }

        self.gop_pos += 1;
        if self.gop_pos >= self.gop_size {
            self.gop_pos = 0;
        }
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.pending.pop_front() {
            return Ok(p);
        }
        if self.eof && !self.finalised {
            // On flush, if we have leftover B-frames in the reorder buffer
            // (trailing Bs of the final GOP with no next anchor yet), promote
            // them to P-frames. They just use the current ref as forward.
            self.finalise_trailing_b_as_p()?;
            if let Some(p) = self.pending.pop_front() {
                return Ok(p);
            }
            self.finalised = true;
            let mut bw = BitWriter::new();
            write_start_code(&mut bw, SEQUENCE_END_CODE);
            let bytes = bw.finish();
            let mut pkt = Packet::new(0, self.time_base, bytes);
            pkt.flags.header = true;
            return Ok(pkt);
        }
        if self.eof {
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum PictureKind {
    I,
    P,
    B,
}

/// Classify a display-order GOP position as I/P/B.
///
/// Position 0 is the I-frame. Positions at multiples of `m = num_b_frames+1`
/// (other than 0) are P-frames. Every other position is a B-frame.
///
/// If there is no valid forward reference yet (`ref_valid == false`),
/// the frame falls back to I (bootstrap).
fn picture_kind_for_position(pos: u32, num_b_frames: u32, ref_valid: bool) -> PictureKind {
    if pos == 0 || !ref_valid {
        return PictureKind::I;
    }
    let m = num_b_frames + 1;
    if pos % m == 0 {
        PictureKind::P
    } else {
        PictureKind::B
    }
}

impl Mpeg1VideoEncoder {
    /// Emit any buffered B-frames, one packet each, using prev_ref as forward
    /// and ref as backward reference. References are assumed up to date for
    /// the B-frames we are about to encode (called right after an anchor is
    /// emitted).
    fn flush_buffered_b_frames(&mut self) -> Result<()> {
        if self.b_queue.is_empty() {
            return Ok(());
        }
        if !self.prev_ref_valid || !self.ref_valid {
            return Err(Error::invalid(
                "B-frame flush: missing forward or backward reference",
            ));
        }
        // Drain in display order (same as insertion order).
        while let Some(qf) = self.b_queue.pop_front() {
            let data = encode_b_picture(self, &qf.frame, qf.temporal_reference)?;
            let mut pkt = Packet::new(0, self.time_base, data);
            pkt.pts = qf.frame.pts;
            pkt.dts = qf.frame.pts;
            pkt.flags.keyframe = false;
            self.pending.push_back(pkt);
        }
        Ok(())
    }

    /// Called on flush() to handle any trailing B-frames that never got a
    /// following anchor. They are re-encoded as P-frames against the current
    /// backward reference.
    fn finalise_trailing_b_as_p(&mut self) -> Result<()> {
        while let Some(qf) = self.b_queue.pop_front() {
            let data = encode_anchor_picture(self, &qf.frame, false, qf.temporal_reference)?;
            let mut pkt = Packet::new(0, self.time_base, data);
            pkt.pts = qf.frame.pts;
            pkt.dts = qf.frame.pts;
            pkt.flags.keyframe = false;
            self.pending.push_back(pkt);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Picture encode
// ---------------------------------------------------------------------------

fn encode_anchor_picture(
    enc: &mut Mpeg1VideoEncoder,
    v: &VideoFrame,
    is_intra: bool,
    temporal_reference: u16,
) -> Result<Vec<u8>> {
    let mut bw = BitWriter::with_capacity(8192);

    let mb_w = (enc.width as usize).div_ceil(16);
    let mb_h = (enc.height as usize).div_ceil(16);

    // Emit Sequence + GOP headers at GOP boundaries (i.e. before the I-frame).
    if is_intra {
        write_start_code(&mut bw, SEQUENCE_HEADER_CODE);
        write_sequence_header(
            &mut bw,
            enc.width,
            enc.height,
            enc.frame_rate_code,
            enc.bit_rate,
        );
        if enc.codec == Codec::Mpeg2 {
            write_start_code(&mut bw, EXTENSION_START_CODE);
            write_sequence_extension(&mut bw, &Mpeg2SequenceExt::default());
            bw.align_to_byte();
        }
        write_start_code(&mut bw, GROUP_START_CODE);
        write_gop_header(&mut bw);
    }

    // Picture header.
    write_start_code(&mut bw, PICTURE_START_CODE);
    if is_intra {
        write_picture_header_i(&mut bw, temporal_reference);
    } else {
        write_picture_header_p(&mut bw, temporal_reference);
    }
    if enc.codec == Codec::Mpeg2 {
        write_start_code(&mut bw, EXTENSION_START_CODE);
        // For I-only output the f_codes are unused; spec requires them to
        // be set to 15 ("forbidden") — our parser tolerates any value since
        // the decoder only consults them when the MB actually has an MV.
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
        write_picture_coding_extension(&mut bw, &ext);
        bw.align_to_byte();
    }

    // Allocate the reconstructed picture for this frame so we can use it as
    // the reference for the next P-frame. Macroblock-aligned dims.
    let y_stride = mb_w * 16;
    let c_stride = mb_w * 8;
    let y_h = mb_h * 16;
    let c_h = mb_h * 8;
    let mut recon_y = vec![0u8; y_stride * y_h];
    let mut recon_cb = vec![0u8; c_stride * c_h];
    let mut recon_cr = vec![0u8; c_stride * c_h];

    // Slices.
    for row in 0..mb_h {
        write_start_code(&mut bw, (row + 1) as u8);
        if is_intra {
            encode_slice_i(
                &mut bw,
                enc,
                v,
                row,
                mb_w,
                &mut recon_y,
                &mut recon_cb,
                &mut recon_cr,
                y_stride,
                c_stride,
            )?;
        } else {
            encode_slice_p(
                &mut bw,
                enc,
                v,
                row,
                mb_w,
                &mut recon_y,
                &mut recon_cb,
                &mut recon_cr,
                y_stride,
                c_stride,
            )?;
        }
    }

    // Roll reference pictures: old backward ref → forward ref, newly
    // reconstructed → backward ref. The forward ref is needed for encoding
    // any buffered B-frames that are display-ordered between the previous
    // anchor and this one.
    if enc.ref_valid {
        enc.prev_ref_y = std::mem::take(&mut enc.ref_y);
        enc.prev_ref_cb = std::mem::take(&mut enc.ref_cb);
        enc.prev_ref_cr = std::mem::take(&mut enc.ref_cr);
        enc.prev_ref_y_stride = enc.ref_y_stride;
        enc.prev_ref_c_stride = enc.ref_c_stride;
        enc.prev_ref_valid = true;
    }
    enc.ref_y = recon_y;
    enc.ref_cb = recon_cb;
    enc.ref_cr = recon_cr;
    enc.ref_y_stride = y_stride;
    enc.ref_c_stride = c_stride;
    enc.ref_valid = true;

    let _ = temporal_reference;
    Ok(bw.finish())
}

/// Encode one B-frame. Does NOT roll reference pictures — B-frames are
/// never used as anchors.
fn encode_b_picture(
    enc: &Mpeg1VideoEncoder,
    v: &VideoFrame,
    temporal_reference: u16,
) -> Result<Vec<u8>> {
    let mut bw = BitWriter::with_capacity(8192);

    let mb_w = (enc.width as usize).div_ceil(16);
    let mb_h = (enc.height as usize).div_ceil(16);

    // Picture header.
    write_start_code(&mut bw, PICTURE_START_CODE);
    write_picture_header_b(&mut bw, temporal_reference);

    for row in 0..mb_h {
        write_start_code(&mut bw, (row + 1) as u8);
        encode_slice_b(&mut bw, enc, v, row, mb_w)?;
    }

    Ok(bw.finish())
}

fn write_start_code(bw: &mut BitWriter, code: u8) {
    bw.align_to_byte();
    bw.write_bytes(&[0x00, 0x00, 0x01, code]);
}

fn write_sequence_header(
    bw: &mut BitWriter,
    width: u32,
    height: u32,
    frame_rate_code: u8,
    bit_rate: u64,
) {
    bw.write_bits(width, 12);
    bw.write_bits(height, 12);
    bw.write_bits(1, 4); // aspect_ratio_info = 1 (square)
    bw.write_bits(frame_rate_code as u32, 4);
    let br_units = bit_rate.div_ceil(400).min(0x3FFFF) as u32;
    bw.write_bits(br_units, 18);
    bw.write_bits(1, 1); // marker
    bw.write_bits(20, 10); // vbv_buffer_size
    bw.write_bits(0, 1); // constrained_parameters_flag
    bw.write_bits(0, 1); // load_intra_quantiser_matrix
    bw.write_bits(0, 1); // load_non_intra_quantiser_matrix
    bw.align_to_byte();
}

fn write_gop_header(bw: &mut BitWriter) {
    bw.write_bits(0, 1); // drop_frame_flag
    bw.write_bits(0, 5); // hours
    bw.write_bits(0, 6); // minutes
    bw.write_bits(1, 1); // marker
    bw.write_bits(0, 6); // seconds
    bw.write_bits(0, 6); // pictures
    bw.write_bits(1, 1); // closed_gop
    bw.write_bits(0, 1); // broken_link
    bw.align_to_byte();
}

fn write_picture_header_i(bw: &mut BitWriter, temporal_reference: u16) {
    bw.write_bits(temporal_reference as u32 & 0x3FF, 10);
    bw.write_bits(1, 3); // picture_coding_type = 1 (I)
    bw.write_bits(0xFFFF, 16); // vbv_delay
    bw.write_bits(0, 1); // extra_bit_picture
    bw.align_to_byte();
}

fn write_picture_header_p(bw: &mut BitWriter, temporal_reference: u16) {
    bw.write_bits(temporal_reference as u32 & 0x3FF, 10);
    bw.write_bits(2, 3); // picture_coding_type = 2 (P)
    bw.write_bits(0xFFFF, 16); // vbv_delay
    bw.write_bits(0, 1); // full_pel_forward_vector = 0
    bw.write_bits(1, 3); // forward_f_code = 1 → ±16 half-pel
    bw.write_bits(0, 1); // extra_bit_picture
    bw.align_to_byte();
}

fn write_picture_header_b(bw: &mut BitWriter, temporal_reference: u16) {
    bw.write_bits(temporal_reference as u32 & 0x3FF, 10);
    bw.write_bits(3, 3); // picture_coding_type = 3 (B)
    bw.write_bits(0xFFFF, 16); // vbv_delay
    bw.write_bits(0, 1); // full_pel_forward_vector = 0
    bw.write_bits(1, 3); // forward_f_code = 1 → ±16 half-pel
    bw.write_bits(0, 1); // full_pel_backward_vector = 0
    bw.write_bits(1, 3); // backward_f_code = 1 → ±16 half-pel
    bw.write_bits(0, 1); // extra_bit_picture
    bw.align_to_byte();
}

// ---------------------------------------------------------------------------
// I-picture slice / MB encode
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn encode_slice_i(
    bw: &mut BitWriter,
    enc: &Mpeg1VideoEncoder,
    v: &VideoFrame,
    mb_row: usize,
    mb_w: usize,
    recon_y: &mut [u8],
    recon_cb: &mut [u8],
    recon_cr: &mut [u8],
    y_stride: usize,
    c_stride: usize,
) -> Result<()> {
    bw.write_bits(enc.quant_scale as u32, 5);
    bw.write_bits(0, 1); // extra_bit_slice

    let mut dc_pred_q: [i32; 3] = [128, 128, 128];

    for mb_col in 0..mb_w {
        // macroblock_address_increment = 1
        bw.write_bits(0b1, 1);
        // macroblock_type for I-picture: `1` (1 bit) = Intra (no quant).
        bw.write_bits(0b1, 1);

        encode_mb_intra(
            bw,
            enc,
            v,
            mb_row,
            mb_col,
            &mut dc_pred_q,
            recon_y,
            recon_cb,
            recon_cr,
            y_stride,
            c_stride,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_mb_intra(
    bw: &mut BitWriter,
    enc: &Mpeg1VideoEncoder,
    v: &VideoFrame,
    mb_row: usize,
    mb_col: usize,
    dc_pred_q: &mut [i32; 3],
    recon_y: &mut [u8],
    recon_cb: &mut [u8],
    recon_cr: &mut [u8],
    y_stride: usize,
    c_stride: usize,
) -> Result<()> {
    let q = enc.quant_scale as i32;
    let intra_q = &DEFAULT_INTRA_QUANT;

    let w = v.width as usize;
    let h = v.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);

    let y_plane = &v.planes[0];
    let cb_plane = &v.planes[1];
    let cr_plane = &v.planes[2];

    let y0 = mb_row * 16;
    let x0 = mb_col * 16;
    let cy0 = mb_row * 8;
    let cx0 = mb_col * 8;

    // 4 luma blocks
    for (bx, by) in [(0usize, 0usize), (8, 0), (0, 8), (8, 8)].iter() {
        encode_block_intra(
            bw,
            &y_plane.data,
            y_plane.stride,
            w,
            h,
            x0 + bx,
            y0 + by,
            false,
            q,
            intra_q,
            &mut dc_pred_q[0],
            recon_y,
            y_stride,
            x0 + bx,
            y0 + by,
            enc.codec,
        )?;
    }
    // Cb
    encode_block_intra(
        bw,
        &cb_plane.data,
        cb_plane.stride,
        cw,
        ch,
        cx0,
        cy0,
        true,
        q,
        intra_q,
        &mut dc_pred_q[1],
        recon_cb,
        c_stride,
        cx0,
        cy0,
        enc.codec,
    )?;
    // Cr
    encode_block_intra(
        bw,
        &cr_plane.data,
        cr_plane.stride,
        cw,
        ch,
        cx0,
        cy0,
        true,
        q,
        intra_q,
        &mut dc_pred_q[2],
        recon_cr,
        c_stride,
        cx0,
        cy0,
        enc.codec,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_block_intra(
    bw: &mut BitWriter,
    plane: &[u8],
    stride: usize,
    pw: usize,
    ph: usize,
    x0: usize,
    y0: usize,
    is_chroma: bool,
    q: i32,
    intra_q: &[u8; 64],
    prev_dc_q: &mut i32,
    recon: &mut [u8],
    recon_stride: usize,
    rx0: usize,
    ry0: usize,
    codec: Codec,
) -> Result<()> {
    // 1. Pull samples (with edge replication).
    let mut samples = [0.0f32; 64];
    for j in 0..8 {
        let yy = (y0 + j).min(ph.saturating_sub(1));
        for i in 0..8 {
            let xx = (x0 + i).min(pw.saturating_sub(1));
            samples[j * 8 + i] = plane[yy * stride + xx] as f32;
        }
    }

    // 2. Forward DCT (no level shift).
    fdct8x8(&mut samples);

    // 3. Quantise. DC step = 8.
    let dc_coeff = samples[0];
    let dc_q = ((dc_coeff / 8.0).round() as i32).clamp(0, 255);
    let dc_diff = dc_q - *prev_dc_q;
    *prev_dc_q = dc_q;

    // 4. Quantise AC coefficients. For intra dequant:
    //    * MPEG-1 spec:  rec = (2 * level * q * W) / 16  →  level ≈ coef * 8 / (q*W)
    //    * MPEG-2 spec:  rec = (level * q * W) / 16      →  level ≈ coef * 16 / (q*W)
    //
    // MPEG-2 uses twice the quantised level for the same coefficient value at
    // the same quant scale. The bit-cost hit is partly offset by the larger
    // useful range (escape is a longer code).
    let mpeg2 = codec == Codec::Mpeg2;
    let quant_mul: f32 = if mpeg2 { 16.0 } else { 8.0 };
    let mut levels = [0i32; 64];
    for k in 1..64 {
        let nat = ZIGZAG[k];
        let coef = samples[nat];
        let qf = intra_q[nat] as f32;
        let denom = q as f32 * qf;
        let v = if denom == 0.0 {
            0.0
        } else {
            coef * quant_mul / denom
        };
        let lv = if v >= 0.0 {
            (v + 0.5) as i32
        } else {
            -(((-v) + 0.5) as i32)
        };
        // MPEG-1: Table B-14 covers |level| ≤ 255 directly (and escape
        //   carries up to ±255 without a long form — actually ±255 max).
        // MPEG-2: 12-bit signed escape carries ±2047 (sans 0 and -2048).
        let limit = if mpeg2 { 2047 } else { 255 };
        levels[k] = lv.clamp(-limit, limit);
    }

    // 5. Encode DC differential.
    encode_dc_diff(bw, dc_diff, is_chroma)?;

    // 6. Encode AC run/level pairs.
    encode_ac_coeffs(bw, &levels, codec)?;

    // 7. Reconstruct (decoder-equivalent dequant + IDCT) into the reference
    //    plane so subsequent P-frames can use it. We also use this to
    //    reconstruct the encoder-side sample for self-test round-trips.
    let mut coeffs = [0i32; 64];
    coeffs[0] = dc_q * 8;
    for k in 1..64 {
        let lv = levels[k];
        if lv == 0 {
            continue;
        }
        let nat = ZIGZAG[k];
        let qf = intra_q[nat] as i32;
        let mut rec = if mpeg2 {
            (lv * q * qf) / 16
        } else {
            (2 * lv * q * qf) / 16
        };
        if !mpeg2 && rec & 1 == 0 && rec != 0 {
            rec = if rec > 0 { rec - 1 } else { rec + 1 };
        }
        rec = rec.clamp(-2048, 2047);
        coeffs[nat] = rec;
    }
    if mpeg2 {
        // H.262 §7.4.4 mismatch: XOR all 64 coefficient LSBs; if the sum is
        // even, flip LSB of coeff[63].
        let mut sum: i32 = 0;
        for &c in coeffs.iter() {
            sum ^= c;
        }
        if sum & 1 == 0 {
            coeffs[63] ^= 1;
            if coeffs[63] == 2048 {
                coeffs[63] = 2047;
            }
            if coeffs[63] == -2049 {
                coeffs[63] = -2048;
            }
        }
    }
    let mut fblock = [0.0f32; 64];
    for i in 0..64 {
        fblock[i] = coeffs[i] as f32;
    }
    idct8x8(&mut fblock);
    for j in 0..8 {
        for i in 0..8 {
            let pix = fblock[j * 8 + i];
            let p = if pix <= 0.0 {
                0
            } else if pix >= 255.0 {
                255
            } else {
                pix.round() as u8
            };
            let dy = ry0 + j;
            let dx = rx0 + i;
            // The reconstructed plane is mb-aligned and not necessarily the
            // same dimension as the picture; clamp to plane bounds.
            recon[dy * recon_stride + dx] = p;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// P-picture slice / MB encode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum PMbMode {
    /// Forward MC, no coded residual (saves CBP + AC bits).
    Forward { mv_x: i32, mv_y: i32 },
    /// Forward MC + coded residual. The actual CBP is computed during
    /// emit (after quantising) since it depends on which blocks have any
    /// surviving nonzero levels.
    ForwardCoded { mv_x: i32, mv_y: i32 },
    /// Intra fallback.
    Intra,
}

#[allow(clippy::too_many_arguments)]
fn encode_slice_p(
    bw: &mut BitWriter,
    enc: &Mpeg1VideoEncoder,
    v: &VideoFrame,
    mb_row: usize,
    mb_w: usize,
    recon_y: &mut [u8],
    recon_cb: &mut [u8],
    recon_cr: &mut [u8],
    y_stride: usize,
    c_stride: usize,
) -> Result<()> {
    bw.write_bits(enc.quant_scale as u32, 5);
    bw.write_bits(0, 1); // extra_bit_slice

    // DC predictors reset at slice start (intra MBs only).
    let mut dc_pred_q: [i32; 3] = [128, 128, 128];
    // MV predictor reset at slice start (and on intra/skip MBs).
    let mut mv_pred = (0i32, 0i32);

    // Pre-compute MB decisions for the whole row so we can collapse runs of
    // skipped MBs into a single MBA increment.
    let mut decisions: Vec<PMbMode> = Vec::with_capacity(mb_w);
    let mut sad_intra: Vec<u32> = Vec::with_capacity(mb_w);
    let _ = (&dc_pred_q, &mv_pred);

    // Phase 1: per-MB ME + decision (without bitstream MV diff cost — we
    // handle the predictor bookkeeping during phase 2).
    for mb_col in 0..mb_w {
        let (best_mv, sad_mc, sad_zero, sad_intra_v) = mb_motion_search(enc, v, mb_row, mb_col);
        let decision = pick_mb_mode_p(best_mv, sad_mc, sad_zero, sad_intra_v);
        decisions.push(decision);
        sad_intra.push(sad_intra_v);
    }

    // Phase 2: emit. We process MBs left-to-right, queuing skip-eligible
    // MBs into a "skip run" that is flushed by emitting a single
    // macroblock_address_increment when the next non-skip MB arrives.
    let mut pending_skip: u32 = 0; // count of MBs currently being skipped
    let mut first_mb_emitted = false;

    for mb_col in 0..mb_w {
        let mode = decisions[mb_col];
        let is_skip = matches!(mode, PMbMode::Forward { mv_x: 0, mv_y: 0 });

        // The first MB in a slice cannot be skipped per spec: its MBA
        // increment must be 1. Force the first MB to be emitted as a coded
        // MB even if its mode would be skip — we promote it to "Forward
        // (0,0)" without skip bookkeeping (that's the same behaviour).
        let force_emit = !first_mb_emitted;

        if is_skip && !force_emit {
            pending_skip += 1;
            continue;
        }

        // Emit the macroblock_address_increment: 1 + pending_skip MBs since
        // the previous emitted (or row start). MBA encoding: write `incr`
        // using Table B-1, possibly preceded by escapes (33-incr) for big
        // gaps.
        let incr = pending_skip + 1;
        write_mba(bw, incr)?;
        pending_skip = 0;

        // Reset MV predictor on skip-runs (per §2.4.4.2: skipped MBs zero
        // their MVs and reset predictors). The "force_emit first MB" path
        // doesn't have a preceding skip, so no reset needed.
        if incr > 1 {
            mv_pred = (0, 0);
        }

        match mode {
            PMbMode::Forward { mv_x, mv_y } => {
                if mv_x == 0 && mv_y == 0 {
                    // The first MB of a slice forced to emit with MV (0,0):
                    // we encode this as "MC, Coded" with CBP = 0? No — CBP
                    // = 0 isn't representable. Fall back to "MC, Not Coded"
                    // which permits no residual.
                    // macroblock_type "001" = MC, Not Coded (forward, no
                    // pattern). This is 3 bits.
                    write_mb_type(bw, MbTypeCode::McNotCoded)?;
                    encode_mv_diff(bw, &mut mv_pred, 0, 0)?;
                    // Reconstruct prediction = previous MB at this position
                    // (zero MV).
                    apply_p_forward_no_residual(
                        enc, mb_col, mb_row, 0, 0, recon_y, recon_cb, recon_cr, y_stride, c_stride,
                    )?;
                    // DC predictor reset (intra predictor only matters for
                    // intra MBs).
                    dc_pred_q = [128, 128, 128];
                } else {
                    // MC, Not Coded (forward only, no pattern). Code "001".
                    write_mb_type(bw, MbTypeCode::McNotCoded)?;
                    encode_mv_diff(bw, &mut mv_pred, mv_x, mv_y)?;
                    apply_p_forward_no_residual(
                        enc, mb_col, mb_row, mv_x, mv_y, recon_y, recon_cb, recon_cr, y_stride,
                        c_stride,
                    )?;
                    dc_pred_q = [128, 128, 128];
                }
            }
            PMbMode::ForwardCoded { mv_x, mv_y } => {
                // First quantise the residual to compute the actual CBP. If
                // CBP comes out 0, demote to MC-Not-Coded so the bitstream
                // stays well-formed (CBP=0 is not representable).
                let block_levels = quantise_p_mb_residual(enc, v, mb_row, mb_col, mv_x, mv_y);
                let mut cbp_actual: u8 = 0;
                for (b, lv) in block_levels.iter().enumerate() {
                    if lv.iter().any(|&l| l != 0) {
                        cbp_actual |= 1 << (5 - b);
                    }
                }
                if cbp_actual == 0 {
                    // Emit as MC-Not-Coded.
                    write_mb_type(bw, MbTypeCode::McNotCoded)?;
                    encode_mv_diff(bw, &mut mv_pred, mv_x, mv_y)?;
                    apply_p_forward_no_residual(
                        enc, mb_col, mb_row, mv_x, mv_y, recon_y, recon_cb, recon_cr, y_stride,
                        c_stride,
                    )?;
                } else {
                    write_mb_type(bw, MbTypeCode::McCoded)?;
                    encode_mv_diff(bw, &mut mv_pred, mv_x, mv_y)?;
                    write_cbp(bw, cbp_actual)?;
                    encode_p_mb_inter_residual_with_levels(
                        bw,
                        enc,
                        mb_row,
                        mb_col,
                        mv_x,
                        mv_y,
                        cbp_actual,
                        &block_levels,
                        recon_y,
                        recon_cb,
                        recon_cr,
                        y_stride,
                        c_stride,
                    )?;
                }
                dc_pred_q = [128, 128, 128];
            }
            PMbMode::Intra => {
                // Intra (5 bits code "00011").
                write_mb_type(bw, MbTypeCode::Intra)?;
                // Spec: when an intra MB appears in a P-picture, the MV
                // predictor is reset to 0.
                mv_pred = (0, 0);
                encode_mb_intra(
                    bw,
                    enc,
                    v,
                    mb_row,
                    mb_col,
                    &mut dc_pred_q,
                    recon_y,
                    recon_cb,
                    recon_cr,
                    y_stride,
                    c_stride,
                )?;
            }
        }

        first_mb_emitted = true;
    }

    // If the row ended on a run of skipped MBs, the last emitted MB became
    // the slice tail and the trailing skip run is intentionally not
    // signalled — the decoder will infer them from the start of the next
    // slice. But that's wrong for the *last* slice of a row! Actually per
    // §2.4.3.1, every MB in the slice must be accounted for — if the last
    // MBs are skipped, they remain implied "not present" and the decoder's
    // termination condition (no more start codes following) will treat
    // them as such. Modern decoders fill them with previous MB or zero MV.
    // To keep our own decoder happy we also need to ensure mb_addr reaches
    // the end. Easiest fix: convert any tail-run skip into MC-Not-Coded
    // emissions so the slice covers every MB explicitly.
    while pending_skip > 0 {
        // Emit a MC-Not-Coded MB with MV=(0,0) for each tailing skipped MB.
        write_mba(bw, 1)?;
        write_mb_type(bw, MbTypeCode::McNotCoded)?;
        encode_mv_diff(bw, &mut mv_pred, 0, 0)?;
        // Reconstruct prediction from reference at this position.
        // tail-fill mb_col index = mb_w - pending_skip.
        let mb_col = mb_w - pending_skip as usize;
        apply_p_forward_no_residual(
            enc, mb_col, mb_row, 0, 0, recon_y, recon_cb, recon_cr, y_stride, c_stride,
        )?;
        pending_skip -= 1;
        let _ = dc_pred_q;
    }

    let _ = sad_intra;
    Ok(())
}

#[derive(Clone, Copy)]
enum MbTypeCode {
    /// "1" — MC, Coded (forward + pattern).
    McCoded,
    /// "01" — No MC, Coded (pattern). Not used by us today.
    #[allow(dead_code)]
    NoMcCoded,
    /// "001" — MC, Not Coded (forward, no pattern).
    McNotCoded,
    /// "00011" — Intra.
    Intra,
}

fn write_mb_type(bw: &mut BitWriter, kind: MbTypeCode) -> Result<()> {
    let (bits, code) = match kind {
        MbTypeCode::McCoded => (1u32, 0b1u32),
        MbTypeCode::NoMcCoded => (2, 0b01),
        MbTypeCode::McNotCoded => (3, 0b001),
        MbTypeCode::Intra => (5, 0b00011),
    };
    // Sanity: lookup the equivalent VLC entry to make sure the table agrees.
    let _ = mb_type::p_table();
    bw.write_bits(code, bits);
    Ok(())
}

fn write_cbp(bw: &mut BitWriter, cbp: u8) -> Result<()> {
    // Look up the coded_block_pattern VLC. cbp=0 isn't representable.
    if cbp == 0 {
        return Err(Error::invalid("encode_cbp: cbp=0"));
    }
    let tbl = cbp_tbl::table();
    let entry =
        lookup_value(tbl, cbp).ok_or_else(|| Error::invalid("CBP value missing in VLC table"))?;
    bw.write_bits(entry.code, entry.bits as u32);
    Ok(())
}

/// Write `incr` using Table B-1. Supports incr ≥ 1 and uses the macroblock
/// escape code (`0000 0001 000`, value 33) for big jumps.
fn write_mba(bw: &mut BitWriter, mut incr: u32) -> Result<()> {
    if incr == 0 {
        return Err(Error::invalid("MBA increment must be ≥ 1"));
    }
    let tbl = mba::table();
    while incr > 33 {
        // Write the escape code.
        let esc =
            lookup_value(tbl, mba::ESCAPE).ok_or_else(|| Error::invalid("MBA escape missing"))?;
        bw.write_bits(esc.code, esc.bits as u32);
        incr -= 33;
    }
    let entry =
        lookup_value(tbl, incr as u8).ok_or_else(|| Error::invalid("MBA value not in table"))?;
    bw.write_bits(entry.code, entry.bits as u32);
    Ok(())
}

// ---------------------------------------------------------------------------
// Motion estimation + decision
// ---------------------------------------------------------------------------

/// Search range in integer pels for forward ME. ±8 covers the spec range
/// |motion_code| ≤ 16 with f_code=1 (16 half-pel = 8 integer pel). The ME
/// adds a small SAD bias proportional to |MV| so that nearly-static MBs
/// are encoded as MV=(0,0), which favours skips. Half-pel refinement below
/// is only attempted when the integer-pel result is within ±7 int pels so
/// the refined |mv| never exceeds 15 half-pel (inside ±16).
const ME_RANGE_PEL: i32 = 8;

/// Full-search block matching with half-pel refinement. Returns:
///   * (best_mv_x, best_mv_y) in **half-pel** units. Integer-pel searches
///     yield even values; the half-pel refinement stage may add ±1 so the
///     returned vector can be odd.
///   * SAD at best MV (computed with half-pel bilinear interpolation where
///     applicable — i.e. the SAD is comparable between integer and
///     fractional candidates).
///   * SAD at MV (0,0) — used to decide the "true skip" case.
///   * SAD as if intra (rough estimate using only luma sample variance).
fn mb_motion_search(
    enc: &Mpeg1VideoEncoder,
    v: &VideoFrame,
    mb_row: usize,
    mb_col: usize,
) -> ((i32, i32), u32, u32, u32) {
    if !enc.ref_valid {
        return ((0, 0), u32::MAX, u32::MAX, u32::MAX);
    }
    let y_plane = &v.planes[0];
    let w = v.width as i32;
    let h = v.height as i32;

    let x0 = (mb_col * 16) as i32;
    let y0 = (mb_row * 16) as i32;
    let mut cur = [0i32; 16 * 16];
    for j in 0..16 {
        for i in 0..16 {
            let xx = (x0 + i).clamp(0, w - 1);
            let yy = (y0 + j).clamp(0, h - 1);
            cur[(j as usize) * 16 + i as usize] =
                y_plane.data[(yy as usize) * y_plane.stride + xx as usize] as i32;
        }
    }

    let ref_y = &enc.ref_y;
    let rs = enc.ref_y_stride as i32;
    let rh = (enc.ref_y.len() / enc.ref_y_stride) as i32;

    // Integer-pel SAD: reference patch at (x0+dx, y0+dy) with edge clamp.
    let sad_int_at = |dx: i32, dy: i32| -> u32 {
        let mut sum: u32 = 0;
        for j in 0..16i32 {
            for i in 0..16i32 {
                let xx = (x0 + i + dx).clamp(0, rs - 1);
                let yy = (y0 + j + dy).clamp(0, rh - 1);
                let r = ref_y[(yy as usize) * (rs as usize) + xx as usize] as i32;
                let c = cur[(j as usize) * 16 + i as usize];
                sum += (c - r).unsigned_abs();
            }
        }
        sum
    };

    // Half-pel SAD using `motion::predict_block` to build the bilinear-
    // interpolated reference patch — matches what the decoder will
    // reconstruct exactly.
    let sad_half_at = |mv_x_half: i32, mv_y_half: i32| -> u32 {
        let mut pred = [0u8; 16 * 16];
        crate::motion::predict_block(
            ref_y,
            enc.ref_y_stride,
            rs,
            rh,
            x0,
            y0,
            mv_x_half,
            mv_y_half,
            16,
            &mut pred,
            16,
        );
        let mut sum: u32 = 0;
        for j in 0..16 {
            for i in 0..16 {
                let c = cur[j * 16 + i];
                let r = pred[j * 16 + i] as i32;
                sum += (c - r).unsigned_abs();
            }
        }
        sum
    };

    // Integer-pel full search. Track (mv_int_x, mv_int_y) and SAD here —
    // convert to half-pel only at the end.
    let mut best_int: ((i32, i32), u32) = ((0, 0), sad_int_at(0, 0));
    let sad_zero = best_int.1;
    // Bias factor: cost in SAD units we charge per unit of |MV| (in integer-
    // pel units). Higher = stronger preference for MV=(0,0). We need decisive
    // wins from MC because each non-zero MV costs ≥ 11 bits in the bitstream
    // and adds quantisation error chains.
    let bias_per_pel: u32 = 16;
    for dy in -ME_RANGE_PEL..=ME_RANGE_PEL {
        for dx in -ME_RANGE_PEL..=ME_RANGE_PEL {
            let s = sad_int_at(dx, dy);
            let bias = (dx.unsigned_abs() + dy.unsigned_abs()) * bias_per_pel;
            let best_bias =
                (best_int.0 .0.unsigned_abs() + best_int.0 .1.unsigned_abs()) * bias_per_pel;
            if s + bias < best_int.1 + best_bias {
                best_int = ((dx, dy), s);
            }
        }
    }
    // Convert integer-pel best to half-pel units.
    let mut best: ((i32, i32), u32) = ((best_int.0 .0 * 2, best_int.0 .1 * 2), best_int.1);

    // Half-pel refinement: test the 8 neighbours of the best integer-pel MV
    // at ±1 half-pel offsets using bilinear-interpolated reference patches
    // (exactly what the decoder will reconstruct). Stay within |mv| ≤ 15
    // half-pel (one below the ±16 spec limit).
    //
    // Half-pel prediction bilinearly smooths the reference; for truly
    // integer-pel motion this is a net loss. Apply a "win threshold" so a
    // fractional candidate only gets picked when it wins by a meaningful
    // margin — avoiding drift accumulation on integer-motion content.
    let bias_per_half: u32 = 8;
    let half_pel_win: u32 = 32;
    let (mut bx, mut by) = best.0;
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let mx = bx + dx;
            let my = by + dy;
            if mx.abs() > 15 || my.abs() > 15 {
                continue;
            }
            let s = sad_half_at(mx, my);
            let bias = (mx.unsigned_abs() + my.unsigned_abs()) * bias_per_half;
            let best_bias = (bx.unsigned_abs() + by.unsigned_abs()) * bias_per_half;
            if s + bias + half_pel_win < best.1 + best_bias {
                best = ((mx, my), s);
                bx = mx;
                by = my;
            }
        }
    }

    // Intra "cost" estimate: mean abs deviation × 16×16 (poor man's
    // variance proxy). Used only for the intra-vs-inter decision.
    let mut mean: i32 = 0;
    for c in cur.iter() {
        mean += c;
    }
    mean /= 256;
    let mut intra_dev: u32 = 0;
    for c in cur.iter() {
        intra_dev += (*c - mean).unsigned_abs();
    }

    (best.0, best.1, sad_zero, intra_dev)
}

fn pick_mb_mode_p(best_mv: (i32, i32), sad_mc: u32, sad_zero: u32, sad_intra: u32) -> PMbMode {
    // Intra fallback: only if the inter SAD is dramatically larger than
    // intra.
    if sad_mc > sad_intra * 3 + 4096 {
        return PMbMode::Intra;
    }
    // True-skip case: MV=(0,0) AND the prediction is bit-identical (SAD
    // = 0). This typically only fires for the constant-flat areas of the
    // testsrc background. Anything else gets ForwardCoded so the residual
    // can correct prediction error introduced by f32 IDCT drift.
    if best_mv == (0, 0) && sad_zero == 0 {
        return PMbMode::Forward { mv_x: 0, mv_y: 0 };
    }
    // Default: emit forward + coded residual. The actual CBP is computed
    // during residual encode and may demote to MC-Not-Coded if quantisation
    // kills every block.
    PMbMode::ForwardCoded {
        mv_x: best_mv.0,
        mv_y: best_mv.1,
    }
}

// ---------------------------------------------------------------------------
// Forward-predicted MB without residual (or skip): just copy MC prediction.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn apply_p_forward_no_residual(
    enc: &Mpeg1VideoEncoder,
    mb_col: usize,
    mb_row: usize,
    mv_x: i32,
    mv_y: i32,
    recon_y: &mut [u8],
    recon_cb: &mut [u8],
    recon_cr: &mut [u8],
    y_stride: usize,
    c_stride: usize,
) -> Result<()> {
    if !enc.ref_valid {
        return Err(Error::invalid("P MB without reference picture"));
    }
    let mut pred_y = [0u8; 16 * 16];
    let mut pred_cb = [0u8; 8 * 8];
    let mut pred_cr = [0u8; 8 * 8];
    build_mc_prediction(
        enc,
        mb_col,
        mb_row,
        mv_x,
        mv_y,
        &mut pred_y,
        &mut pred_cb,
        &mut pred_cr,
    );
    // Write into reconstructed plane.
    let yx = mb_col * 16;
    let yy = mb_row * 16;
    for j in 0..16 {
        let dst_off = (yy + j) * y_stride + yx;
        recon_y[dst_off..dst_off + 16].copy_from_slice(&pred_y[j * 16..j * 16 + 16]);
    }
    let cx = mb_col * 8;
    let cy = mb_row * 8;
    for j in 0..8 {
        let dst_off = (cy + j) * c_stride + cx;
        recon_cb[dst_off..dst_off + 8].copy_from_slice(&pred_cb[j * 8..j * 8 + 8]);
        recon_cr[dst_off..dst_off + 8].copy_from_slice(&pred_cr[j * 8..j * 8 + 8]);
    }
    Ok(())
}

/// Build 16x16 luma + 8x8 chroma prediction from the encoder's reference
/// picture. Mirrors `crate::motion::predict_block` so the decoder will see
/// the same prediction.
#[allow(clippy::too_many_arguments)]
fn build_mc_prediction(
    enc: &Mpeg1VideoEncoder,
    mb_col: usize,
    mb_row: usize,
    mv_x: i32,
    mv_y: i32,
    pred_y: &mut [u8; 16 * 16],
    pred_cb: &mut [u8; 8 * 8],
    pred_cr: &mut [u8; 8 * 8],
) {
    let mb_px = (mb_col * 16) as i32;
    let mb_py = (mb_row * 16) as i32;
    let ry_h = (enc.ref_y.len() / enc.ref_y_stride) as i32;
    crate::motion::predict_block(
        &enc.ref_y,
        enc.ref_y_stride,
        enc.ref_y_stride as i32,
        ry_h,
        mb_px,
        mb_py,
        mv_x,
        mv_y,
        16,
        pred_y,
        16,
    );
    let c_px = (mb_col * 8) as i32;
    let c_py = (mb_row * 8) as i32;
    let mv_cx = crate::motion::scale_mv_to_chroma(mv_x);
    let mv_cy = crate::motion::scale_mv_to_chroma(mv_y);
    let rc_h = (enc.ref_cb.len() / enc.ref_c_stride) as i32;
    crate::motion::predict_block(
        &enc.ref_cb,
        enc.ref_c_stride,
        enc.ref_c_stride as i32,
        rc_h,
        c_px,
        c_py,
        mv_cx,
        mv_cy,
        8,
        pred_cb,
        8,
    );
    crate::motion::predict_block(
        &enc.ref_cr,
        enc.ref_c_stride,
        enc.ref_c_stride as i32,
        rc_h,
        c_px,
        c_py,
        mv_cx,
        mv_cy,
        8,
        pred_cr,
        8,
    );
}

// ---------------------------------------------------------------------------
// MV differential VLC encoding (Table B-10).
// ---------------------------------------------------------------------------

/// Encode the forward MV (mv_x, mv_y) for the current MB given the running
/// predictor `pred`. `mv_x, mv_y` are in half-pel units (any even value
/// since we're integer-pel only). With `forward_f_code = 1` (f=1), the
/// reconstructed-vector range is [-32, 31] half-pel and complement_r is
/// 0 bits.
fn encode_mv_diff(bw: &mut BitWriter, pred: &mut (i32, i32), mv_x: i32, mv_y: i32) -> Result<()> {
    encode_one_mv_component(bw, &mut pred.0, mv_x)?;
    encode_one_mv_component(bw, &mut pred.1, mv_y)?;
    Ok(())
}

fn encode_one_mv_component(bw: &mut BitWriter, predictor: &mut i32, target: i32) -> Result<()> {
    // Range for f_code=1: [-32, 31]. complement_r is 0 bits because f=1.
    let f: i32 = 1;
    let range: i32 = 32 * f;

    // Per spec the decoder reconstructs:
    //   new = predictor + sign(motion_code) * little
    //   little = (|motion_code| - 1) * f + complement_r + 1   → for f=1, = |motion_code|
    //   new is then wrapped into [-range, range-1]
    // To target a specific reconstructed value `target`, we need to pick a
    // delta = motion_code such that ((predictor + delta) wrapped) == target.
    // delta candidates: target - predictor, ±64.
    let raw = target - *predictor;
    let candidates = [raw, raw + 2 * range, raw - 2 * range];
    let mut chosen: Option<i32> = None;
    for d in candidates {
        if d.abs() <= MAX_MOTION_CODE {
            chosen = Some(d);
            break;
        }
    }
    // If the requested target isn't representable from the current predictor,
    // pick the representable delta closest to `raw` (clamped to
    // ±MAX_MOTION_CODE). This is still a lossy choice — the caller's
    // motion-compensated prediction will not match what the decoder
    // reconstructs — so we update the predictor to the *actual*
    // reconstructed MV and rely on the caller having already committed the
    // chosen MV upstream. In practice this path is rare because ME clamps
    // |mv| to ±15 half-pel and skip-runs reset the predictor.
    let delta = chosen.unwrap_or_else(|| raw.clamp(-MAX_MOTION_CODE, MAX_MOTION_CODE));

    // motion_code = delta. abs(delta) is the table value (0..=16); sign goes
    // separately when nonzero.
    let abs = delta.unsigned_abs();
    if abs > 16 {
        return Err(Error::invalid("|motion_code| > 16"));
    }
    let entry = lookup_motion_code(abs as u8)
        .ok_or_else(|| Error::invalid("motion_code not in Table B-10"))?;
    bw.write_bits(entry.code, entry.bits as u32);
    if delta != 0 {
        let sign = if delta < 0 { 1 } else { 0 };
        bw.write_bits(sign, 1);
    }
    // f=1 → no complement_r bits.

    // Update predictor to the reconstructed value.
    let new_pred = *predictor + delta;
    let wrapped = if new_pred < -range {
        new_pred + 2 * range
    } else if new_pred > range - 1 {
        new_pred - 2 * range
    } else {
        new_pred
    };
    *predictor = wrapped;
    Ok(())
}

fn lookup_motion_code(abs: u8) -> Option<VlcEntry<u8>> {
    let tbl = mv_tbl::table();
    tbl.entries.iter().find(|e| e.value == abs).copied()
}

// ---------------------------------------------------------------------------
// P-MB inter residual (forward MC + coded residual)
// ---------------------------------------------------------------------------

/// Compute the (per-block, mid-tread quantised) residual levels for an
/// inter macroblock with the given forward MV. Returns a 6×64 array of
/// quantised levels in zigzag order.
fn quantise_p_mb_residual(
    enc: &Mpeg1VideoEncoder,
    v: &VideoFrame,
    mb_row: usize,
    mb_col: usize,
    mv_x: i32,
    mv_y: i32,
) -> [[i32; 64]; 6] {
    let mut out = [[0i32; 64]; 6];
    if !enc.ref_valid {
        return out;
    }
    let mut pred_y = [0u8; 16 * 16];
    let mut pred_cb = [0u8; 8 * 8];
    let mut pred_cr = [0u8; 8 * 8];
    build_mc_prediction(
        enc,
        mb_col,
        mb_row,
        mv_x,
        mv_y,
        &mut pred_y,
        &mut pred_cb,
        &mut pred_cr,
    );

    let q = enc.quant_scale as i32;
    let non_intra_q = &DEFAULT_NON_INTRA_QUANT;

    let w = v.width as usize;
    let h = v.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);

    let y_plane = &v.planes[0];
    let cb_plane = &v.planes[1];
    let cr_plane = &v.planes[2];

    let mb_x_pix = mb_col * 16;
    let mb_y_pix = mb_row * 16;
    let mb_cx_pix = mb_col * 8;
    let mb_cy_pix = mb_row * 8;

    for b in 0..6usize {
        let (plane_data, plane_stride, pw, ph, src_x0, src_y0, pred_slice, pred_stride): (
            &[u8],
            usize,
            usize,
            usize,
            usize,
            usize,
            &[u8],
            usize,
        ) = match b {
            0 => (
                &y_plane.data[..],
                y_plane.stride,
                w,
                h,
                mb_x_pix,
                mb_y_pix,
                &pred_y[0..],
                16,
            ),
            1 => (
                &y_plane.data[..],
                y_plane.stride,
                w,
                h,
                mb_x_pix + 8,
                mb_y_pix,
                &pred_y[8..],
                16,
            ),
            2 => (
                &y_plane.data[..],
                y_plane.stride,
                w,
                h,
                mb_x_pix,
                mb_y_pix + 8,
                &pred_y[16 * 8..],
                16,
            ),
            3 => (
                &y_plane.data[..],
                y_plane.stride,
                w,
                h,
                mb_x_pix + 8,
                mb_y_pix + 8,
                &pred_y[16 * 8 + 8..],
                16,
            ),
            4 => (
                &cb_plane.data[..],
                cb_plane.stride,
                cw,
                ch,
                mb_cx_pix,
                mb_cy_pix,
                &pred_cb[..],
                8,
            ),
            5 => (
                &cr_plane.data[..],
                cr_plane.stride,
                cw,
                ch,
                mb_cx_pix,
                mb_cy_pix,
                &pred_cr[..],
                8,
            ),
            _ => unreachable!(),
        };

        let mut residual = [0.0f32; 64];
        for j in 0..8 {
            let yy = (src_y0 + j).min(ph.saturating_sub(1));
            for i in 0..8 {
                let xx = (src_x0 + i).min(pw.saturating_sub(1));
                let s = plane_data[yy * plane_stride + xx] as i32;
                let p = pred_slice[j * pred_stride + i] as i32;
                residual[j * 8 + i] = (s - p) as f32;
            }
        }
        fdct8x8(&mut residual);
        for k in 0..64 {
            let nat = ZIGZAG[k];
            let coef = residual[nat];
            let qf = non_intra_q[nat] as f32;
            let denom = q as f32 * qf;
            if denom == 0.0 {
                continue;
            }
            let abs_c = coef.abs();
            // Optimal-magnitude quantiser: minimise |c - rec(L)| where
            // rec(L) = sign(c) * (2L+1) * Q*W / 16.
            //   L_opt ≈ |c|*8 / (Q*W) - 0.5
            let l_opt = abs_c * 8.0 / denom - 0.5;
            let l = l_opt.round() as i32;
            let lv = if l <= 0 {
                0
            } else if coef >= 0.0 {
                l
            } else {
                -l
            };
            out[b][k] = lv.clamp(-255, 255);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn encode_p_mb_inter_residual_with_levels(
    bw: &mut BitWriter,
    enc: &Mpeg1VideoEncoder,
    mb_row: usize,
    mb_col: usize,
    mv_x: i32,
    mv_y: i32,
    cbp: u8,
    block_levels: &[[i32; 64]; 6],
    recon_y: &mut [u8],
    recon_cb: &mut [u8],
    recon_cr: &mut [u8],
    y_stride: usize,
    c_stride: usize,
) -> Result<()> {
    // Build prediction.
    let mut pred_y = [0u8; 16 * 16];
    let mut pred_cb = [0u8; 8 * 8];
    let mut pred_cr = [0u8; 8 * 8];
    build_mc_prediction(
        enc,
        mb_col,
        mb_row,
        mv_x,
        mv_y,
        &mut pred_y,
        &mut pred_cb,
        &mut pred_cr,
    );

    let q = enc.quant_scale as i32;
    let non_intra_q = &DEFAULT_NON_INTRA_QUANT;

    let mb_x_pix = mb_col * 16;
    let mb_y_pix = mb_row * 16;
    let mb_cx_pix = mb_col * 8;
    let mb_cy_pix = mb_row * 8;

    for b in 0..6usize {
        let coded = (cbp & (1 << (5 - b))) != 0;
        let (pred_slice, pred_stride, recon_dst_stride, recon_dst, dst_x0, dst_y0): (
            &[u8],
            usize,
            usize,
            &mut [u8],
            usize,
            usize,
        ) = match b {
            0 => (&pred_y[0..], 16, y_stride, recon_y, mb_x_pix, mb_y_pix),
            1 => (&pred_y[8..], 16, y_stride, recon_y, mb_x_pix + 8, mb_y_pix),
            2 => (
                &pred_y[16 * 8..],
                16,
                y_stride,
                recon_y,
                mb_x_pix,
                mb_y_pix + 8,
            ),
            3 => (
                &pred_y[16 * 8 + 8..],
                16,
                y_stride,
                recon_y,
                mb_x_pix + 8,
                mb_y_pix + 8,
            ),
            4 => (&pred_cb[..], 8, c_stride, recon_cb, mb_cx_pix, mb_cy_pix),
            5 => (&pred_cr[..], 8, c_stride, recon_cr, mb_cx_pix, mb_cy_pix),
            _ => unreachable!(),
        };

        if !coded {
            // Just copy prediction into the recon plane.
            for j in 0..8 {
                let dst_off = (dst_y0 + j) * recon_dst_stride + dst_x0;
                recon_dst[dst_off..dst_off + 8]
                    .copy_from_slice(&pred_slice[j * pred_stride..j * pred_stride + 8]);
            }
            continue;
        }

        let levels = &block_levels[b];

        // Emit AC coefficients using non-intra (first coeff special) table.
        encode_non_intra_block(bw, levels)?;

        // Reconstruct sample = clamp(prediction + IDCT(dequant(levels))).
        let mut coeffs = [0i32; 64];
        for k in 0..64 {
            let lv = levels[k];
            if lv == 0 {
                continue;
            }
            let nat = ZIGZAG[k];
            let qf = non_intra_q[nat] as i32;
            let add = if lv > 0 { 1 } else { -1 };
            let mut rec = ((2 * lv + add) * q * qf) / 16;
            if rec & 1 == 0 && rec != 0 {
                rec = if rec > 0 { rec - 1 } else { rec + 1 };
            }
            rec = rec.clamp(-2048, 2047);
            coeffs[nat] = rec;
        }
        let mut fblock = [0.0f32; 64];
        for i in 0..64 {
            fblock[i] = coeffs[i] as f32;
        }
        idct8x8(&mut fblock);
        for j in 0..8 {
            for i in 0..8 {
                let p = pred_slice[j * pred_stride + i] as i32;
                let r = fblock[j * 8 + i].round() as i32;
                let pix = (p + r).clamp(0, 255) as u8;
                let dst_off = (dst_y0 + j) * recon_dst_stride + dst_x0 + i;
                recon_dst[dst_off] = pix;
            }
        }
    }

    Ok(())
}

/// Encode a non-intra block's AC coefficients. The first nonzero coefficient
/// uses the "first-coeff" table interpretation (1s = ±1 level instead of EOB);
/// subsequent ones use the regular Table B-14. The block must contain at least
/// one nonzero level (caller's responsibility). Only MPEG-1 P/B encoders call
/// this fn in the first-pass milestone.
fn encode_non_intra_block(bw: &mut BitWriter, levels: &[i32; 64]) -> Result<()> {
    let codec = Codec::Mpeg1;
    let mut first = true;
    let mut run: u32 = 0;
    for k in 0..64 {
        let lv = levels[k];
        if lv == 0 {
            run += 1;
            continue;
        }
        if first {
            // First nonzero coefficient: code special "1s" if run=0,
            // |lv|=1 — otherwise fall back to the regular run/level VLC
            // with the `RunLevel(0,1)` collision NOT possible (we'd hit
            // the `1s` case instead).
            let abs = lv.unsigned_abs();
            if run == 0 && abs == 1 {
                bw.write_bits(0b1, 1);
                let sign = if lv < 0 { 1 } else { 0 };
                bw.write_bits(sign, 1);
                first = false;
                run = 0;
                continue;
            }
            // Otherwise use the normal table for this run/level (the `0b11`
            // collision encodes (run=0, level=1) as 2-bit code, but since
            // we're in first-coeff mode we know the decoder uses
            // first_coeff_table which excludes EOB, so any 2-bit `11` is
            // unambiguously RunLevel(0,1) — same as the regular table.
            // For all other (run, level) combinations the encoding is
            // identical between first and regular tables.
            if let Some(entry) = lookup_run_level(run, abs) {
                bw.write_bits(entry.code, entry.bits as u32);
                let sign = if lv < 0 { 1 } else { 0 };
                bw.write_bits(sign, 1);
            } else {
                emit_escape(bw, run, lv, codec)?;
            }
            first = false;
            run = 0;
            continue;
        }

        let abs = lv.unsigned_abs();
        if let Some(entry) = lookup_run_level(run, abs) {
            bw.write_bits(entry.code, entry.bits as u32);
            let sign = if lv < 0 { 1 } else { 0 };
            bw.write_bits(sign, 1);
        } else {
            emit_escape(bw, run, lv, codec)?;
        }
        run = 0;
    }
    // EOB.
    let eob = find_eob_entry();
    bw.write_bits(eob.code, eob.bits as u32);
    Ok(())
}

fn emit_escape(bw: &mut BitWriter, run: u32, lv: i32, codec: Codec) -> Result<()> {
    let escape_entry = find_escape_entry();
    bw.write_bits(escape_entry.code, escape_entry.bits as u32);
    bw.write_bits(run, 6);
    match codec {
        Codec::Mpeg1 => {
            if (1..=127).contains(&lv) || (-127..=-1).contains(&lv) {
                let v = lv & 0xFF;
                bw.write_bits(v as u32, 8);
            } else if (128..=255).contains(&lv) {
                bw.write_bits(0, 8);
                bw.write_bits(lv as u32, 8);
            } else if (-255..=-128).contains(&lv) {
                bw.write_bits(0x80, 8);
                bw.write_bits((lv + 256) as u32 & 0xFF, 8);
            } else {
                return Err(Error::invalid("AC level out of MPEG-1 range"));
            }
        }
        Codec::Mpeg2 => {
            if lv == 0 || lv == -2048 || !(-2047..=2047).contains(&lv) {
                return Err(Error::invalid("AC level out of MPEG-2 escape range"));
            }
            // 12-bit two's-complement.
            let bits = (lv & 0xFFF) as u32;
            bw.write_bits(bits, 12);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// VLC encode helpers (shared with I path)
// ---------------------------------------------------------------------------

fn encode_dc_diff(bw: &mut BitWriter, diff: i32, is_chroma: bool) -> Result<()> {
    let abs = diff.unsigned_abs();
    let size: u8 = if abs == 0 {
        0
    } else {
        (32 - abs.leading_zeros()) as u8
    };
    if size > 11 {
        return Err(Error::invalid("DC differential out of range"));
    }
    let dc_tbl = if is_chroma {
        dct_dc::chroma()
    } else {
        dct_dc::luma()
    };
    let entry =
        lookup_value(dc_tbl, size).ok_or_else(|| Error::invalid("DC size missing in VLC"))?;
    bw.write_bits(entry.code, entry.bits as u32);
    if size > 0 {
        let bits = encode_signed_field(diff, size as u32);
        bw.write_bits(bits, size as u32);
    }
    Ok(())
}

fn encode_signed_field(value: i32, size: u32) -> u32 {
    if size == 0 {
        return 0;
    }
    let mask = if size == 32 {
        u32::MAX
    } else {
        (1u32 << size) - 1
    };
    if value >= 0 {
        value as u32 & mask
    } else {
        let max_unsigned = (1u32 << size) - 1;
        ((value + max_unsigned as i32) as u32) & mask
    }
}

fn encode_ac_coeffs(bw: &mut BitWriter, levels: &[i32; 64], codec: Codec) -> Result<()> {
    let mut run: u32 = 0;
    for k in 1..64 {
        let lv = levels[k];
        if lv == 0 {
            run += 1;
            continue;
        }
        let abs = lv.unsigned_abs();
        if let Some(entry) = lookup_run_level(run, abs) {
            bw.write_bits(entry.code, entry.bits as u32);
            let sign = if lv < 0 { 1 } else { 0 };
            bw.write_bits(sign, 1);
        } else {
            emit_escape(bw, run, lv, codec)?;
        }
        run = 0;
    }
    let eob = find_eob_entry();
    bw.write_bits(eob.code, eob.bits as u32);
    Ok(())
}

fn lookup_value<T: Copy + PartialEq>(
    tbl: &crate::vlc::VlcTable<T>,
    needle: T,
) -> Option<VlcEntry<T>> {
    tbl.entries.iter().find(|e| e.value == needle).copied()
}

fn lookup_run_level(run: u32, level_abs: u32) -> Option<VlcEntry<DctSym>> {
    if level_abs == 0 || run > 31 {
        return None;
    }
    let tbl = dct_coeffs::table();
    for e in tbl.entries.iter() {
        if let DctSym::RunLevel {
            run: r,
            level_abs: lv,
        } = e.value
        {
            if r as u32 == run && lv as u32 == level_abs {
                return Some(*e);
            }
        }
    }
    None
}

fn find_escape_entry() -> VlcEntry<DctSym> {
    *dct_coeffs::table()
        .entries
        .iter()
        .find(|e| matches!(e.value, DctSym::Escape))
        .expect("escape entry must exist")
}

fn find_eob_entry() -> VlcEntry<DctSym> {
    *dct_coeffs::table()
        .entries
        .iter()
        .find(|e| matches!(e.value, DctSym::Eob))
        .expect("EOB entry must exist")
}

// ---------------------------------------------------------------------------
// B-picture slice / MB encode
// ---------------------------------------------------------------------------

/// B-frame per-MB coding decision. Every variant carries "no coded residual"
/// (no pattern) — this is a conscious simplification: with two-sided
/// prediction available, the residual is typically small and the B-frame
/// bit budget savings come primarily from the better prediction itself
/// rather than from residual corrections.
#[derive(Clone, Copy, Debug)]
enum BMbMode {
    /// Forward motion only.
    Forward { mv_x: i32, mv_y: i32 },
    /// Backward motion only.
    Backward { mv_x: i32, mv_y: i32 },
    /// Bidirectional: average of forward and backward predictions.
    Interpolated {
        fwd_x: i32,
        fwd_y: i32,
        bwd_x: i32,
        bwd_y: i32,
    },
    /// Intra fallback.
    Intra,
}

/// MB-type VLC codes for B-pictures (Table B-4). These are the "not coded"
/// (no pattern) variants. See `tables::mb_type::B_TABLE`.
#[derive(Clone, Copy)]
enum BMbTypeCode {
    /// `10` — Interpolated, Not Coded (fwd + bwd).
    InterpolatedNotCoded,
    /// `010` — Backward, Not Coded.
    BackwardNotCoded,
    /// `0010` — Forward, Not Coded.
    ForwardNotCoded,
    /// `00011` — Intra.
    Intra,
}

fn write_b_mb_type(bw: &mut BitWriter, kind: BMbTypeCode) {
    let (bits, code) = match kind {
        BMbTypeCode::InterpolatedNotCoded => (2u32, 0b10u32),
        BMbTypeCode::BackwardNotCoded => (3, 0b010),
        BMbTypeCode::ForwardNotCoded => (4, 0b0010),
        BMbTypeCode::Intra => (5, 0b00011),
    };
    bw.write_bits(code, bits);
}

#[allow(clippy::too_many_arguments)]
fn encode_slice_b(
    bw: &mut BitWriter,
    enc: &Mpeg1VideoEncoder,
    v: &VideoFrame,
    mb_row: usize,
    mb_w: usize,
) -> Result<()> {
    if !enc.prev_ref_valid || !enc.ref_valid {
        return Err(Error::invalid(
            "encode_slice_b: missing forward or backward reference",
        ));
    }

    bw.write_bits(enc.quant_scale as u32, 5);
    bw.write_bits(0, 1); // extra_bit_slice

    // Per-MB DC predictor (for intra-fallback blocks) + running MV
    // predictors (one per direction). Both are reset on slice entry,
    // skip-runs, and on intra MBs.
    let mut dc_pred_q: [i32; 3] = [128, 128, 128];
    let mut fwd_pred = (0i32, 0i32);
    let mut bwd_pred = (0i32, 0i32);

    for mb_col in 0..mb_w {
        // Motion search against each reference. The forward reference is
        // `prev_ref_*`, the backward reference is `ref_*`.
        let (fwd_best, fwd_sad, _, intra_dev) =
            motion_search_against(v, mb_row, mb_col, &enc.prev_ref_y, enc.prev_ref_y_stride);
        let (bwd_best, bwd_sad, _, _) =
            motion_search_against(v, mb_row, mb_col, &enc.ref_y, enc.ref_y_stride);

        // Bidirectional SAD: average of the two reference patches.
        let bi_sad = bi_sad_for(
            v,
            mb_row,
            mb_col,
            &enc.prev_ref_y,
            enc.prev_ref_y_stride,
            fwd_best.0,
            fwd_best.1,
            &enc.ref_y,
            enc.ref_y_stride,
            bwd_best.0,
            bwd_best.1,
        );

        // Pick min. Favor smaller-SAD modes; intra only if nothing else
        // comes remotely close.
        let best_inter_sad = fwd_sad.min(bwd_sad).min(bi_sad);
        let mode = if best_inter_sad > intra_dev * 4 + 6000 {
            BMbMode::Intra
        } else if bi_sad <= fwd_sad && bi_sad <= bwd_sad {
            BMbMode::Interpolated {
                fwd_x: fwd_best.0,
                fwd_y: fwd_best.1,
                bwd_x: bwd_best.0,
                bwd_y: bwd_best.1,
            }
        } else if fwd_sad <= bwd_sad {
            BMbMode::Forward {
                mv_x: fwd_best.0,
                mv_y: fwd_best.1,
            }
        } else {
            BMbMode::Backward {
                mv_x: bwd_best.0,
                mv_y: bwd_best.1,
            }
        };

        // The first MB of a slice must be coded explicitly (MBA = 1 and
        // cannot use skip). We emit every B-frame MB explicitly (no skip
        // compression in this first version), so MBA = 1 always.
        write_mba(bw, 1)?;

        match mode {
            BMbMode::Forward { mv_x, mv_y } => {
                write_b_mb_type(bw, BMbTypeCode::ForwardNotCoded);
                encode_mv_diff(bw, &mut fwd_pred, mv_x, mv_y)?;
                // Backward predictor stays untouched (no bwd MV in this MB).
                // DC reset (intra predictor only matters for intra MBs).
                dc_pred_q = [128, 128, 128];
            }
            BMbMode::Backward { mv_x, mv_y } => {
                write_b_mb_type(bw, BMbTypeCode::BackwardNotCoded);
                encode_mv_diff(bw, &mut bwd_pred, mv_x, mv_y)?;
                dc_pred_q = [128, 128, 128];
            }
            BMbMode::Interpolated {
                fwd_x,
                fwd_y,
                bwd_x,
                bwd_y,
            } => {
                write_b_mb_type(bw, BMbTypeCode::InterpolatedNotCoded);
                encode_mv_diff(bw, &mut fwd_pred, fwd_x, fwd_y)?;
                encode_mv_diff(bw, &mut bwd_pred, bwd_x, bwd_y)?;
                dc_pred_q = [128, 128, 128];
            }
            BMbMode::Intra => {
                write_b_mb_type(bw, BMbTypeCode::Intra);
                // Spec: when an intra MB appears in a B-picture, the MV
                // predictors are reset to 0.
                fwd_pred = (0, 0);
                bwd_pred = (0, 0);
                // Encode intra residual. The reconstructed samples are
                // written into a throwaway buffer — B-frames are never
                // used as references so we don't need to keep the
                // reconstruction around.
                encode_mb_intra_throwaway(bw, enc, v, mb_row, mb_col, &mut dc_pred_q)?;
            }
        }
    }

    Ok(())
}

/// Run a full-search motion estimation against an arbitrary reference plane
/// buffer. Same algorithm as `mb_motion_search` but parameterised on which
/// reference we're comparing against.
///
/// Returns `(best_mv, sad_at_best, sad_at_zero, intra_dev)`.
fn motion_search_against(
    v: &VideoFrame,
    mb_row: usize,
    mb_col: usize,
    ref_y: &[u8],
    ref_y_stride: usize,
) -> ((i32, i32), u32, u32, u32) {
    if ref_y.is_empty() {
        return ((0, 0), u32::MAX, u32::MAX, u32::MAX);
    }
    let y_plane = &v.planes[0];
    let w = v.width as i32;
    let h = v.height as i32;

    let x0 = (mb_col * 16) as i32;
    let y0 = (mb_row * 16) as i32;
    let mut cur = [0i32; 16 * 16];
    for j in 0..16 {
        for i in 0..16 {
            let xx = (x0 + i).clamp(0, w - 1);
            let yy = (y0 + j).clamp(0, h - 1);
            cur[(j as usize) * 16 + i as usize] =
                y_plane.data[(yy as usize) * y_plane.stride + xx as usize] as i32;
        }
    }

    let rs = ref_y_stride as i32;
    let rh = (ref_y.len() / ref_y_stride) as i32;

    let sad_int_at = |dx: i32, dy: i32| -> u32 {
        let mut sum: u32 = 0;
        for j in 0..16i32 {
            for i in 0..16i32 {
                let xx = (x0 + i + dx).clamp(0, rs - 1);
                let yy = (y0 + j + dy).clamp(0, rh - 1);
                let r = ref_y[(yy as usize) * (rs as usize) + xx as usize] as i32;
                let c = cur[(j as usize) * 16 + i as usize];
                sum += (c - r).unsigned_abs();
            }
        }
        sum
    };

    let sad_half_at = |mv_x_half: i32, mv_y_half: i32| -> u32 {
        let mut pred = [0u8; 16 * 16];
        crate::motion::predict_block(
            ref_y,
            ref_y_stride,
            rs,
            rh,
            x0,
            y0,
            mv_x_half,
            mv_y_half,
            16,
            &mut pred,
            16,
        );
        let mut sum: u32 = 0;
        for j in 0..16 {
            for i in 0..16 {
                let c = cur[j * 16 + i];
                let r = pred[j * 16 + i] as i32;
                sum += (c - r).unsigned_abs();
            }
        }
        sum
    };

    let mut best_int: ((i32, i32), u32) = ((0, 0), sad_int_at(0, 0));
    let sad_zero = best_int.1;
    let bias_per_pel: u32 = 16;
    for dy in -ME_RANGE_PEL..=ME_RANGE_PEL {
        for dx in -ME_RANGE_PEL..=ME_RANGE_PEL {
            let s = sad_int_at(dx, dy);
            let bias = (dx.unsigned_abs() + dy.unsigned_abs()) * bias_per_pel;
            let best_bias =
                (best_int.0 .0.unsigned_abs() + best_int.0 .1.unsigned_abs()) * bias_per_pel;
            if s + bias < best_int.1 + best_bias {
                best_int = ((dx, dy), s);
            }
        }
    }
    let mut best: ((i32, i32), u32) = ((best_int.0 .0 * 2, best_int.0 .1 * 2), best_int.1);

    let bias_per_half: u32 = 8;
    let half_pel_win: u32 = 32;
    let (mut bx, mut by) = best.0;
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let mx = bx + dx;
            let my = by + dy;
            if mx.abs() > 15 || my.abs() > 15 {
                continue;
            }
            let s = sad_half_at(mx, my);
            let bias = (mx.unsigned_abs() + my.unsigned_abs()) * bias_per_half;
            let best_bias = (bx.unsigned_abs() + by.unsigned_abs()) * bias_per_half;
            if s + bias + half_pel_win < best.1 + best_bias {
                best = ((mx, my), s);
                bx = mx;
                by = my;
            }
        }
    }

    let mut mean: i32 = 0;
    for c in cur.iter() {
        mean += c;
    }
    mean /= 256;
    let mut intra_dev: u32 = 0;
    for c in cur.iter() {
        intra_dev += (*c - mean).unsigned_abs();
    }

    (best.0, best.1, sad_zero, intra_dev)
}

/// Compute the luma SAD of an "interpolated" (bidirectional-averaged) MB
/// prediction against the current picture's MB at (mb_col, mb_row). Mirrors
/// the decoder's reconstruction: take fwd MC patch, take bwd MC patch,
/// average with rounding, compare to the current samples.
#[allow(clippy::too_many_arguments)]
fn bi_sad_for(
    v: &VideoFrame,
    mb_row: usize,
    mb_col: usize,
    fwd_ref: &[u8],
    fwd_stride: usize,
    fwd_mv_x: i32,
    fwd_mv_y: i32,
    bwd_ref: &[u8],
    bwd_stride: usize,
    bwd_mv_x: i32,
    bwd_mv_y: i32,
) -> u32 {
    let y_plane = &v.planes[0];
    let w = v.width as i32;
    let h = v.height as i32;
    let x0 = (mb_col * 16) as i32;
    let y0 = (mb_row * 16) as i32;

    let mut fwd = [0u8; 16 * 16];
    let mut bwd = [0u8; 16 * 16];
    let frs = fwd_stride as i32;
    let frh = (fwd_ref.len() / fwd_stride) as i32;
    let brs = bwd_stride as i32;
    let brh = (bwd_ref.len() / bwd_stride) as i32;
    crate::motion::predict_block(
        fwd_ref, fwd_stride, frs, frh, x0, y0, fwd_mv_x, fwd_mv_y, 16, &mut fwd, 16,
    );
    crate::motion::predict_block(
        bwd_ref, bwd_stride, brs, brh, x0, y0, bwd_mv_x, bwd_mv_y, 16, &mut bwd, 16,
    );

    let mut sum: u32 = 0;
    for j in 0..16i32 {
        for i in 0..16i32 {
            let xx = (x0 + i).clamp(0, w - 1);
            let yy = (y0 + j).clamp(0, h - 1);
            let c = y_plane.data[(yy as usize) * y_plane.stride + xx as usize] as i32;
            let f = fwd[(j as usize) * 16 + i as usize] as u32;
            let b = bwd[(j as usize) * 16 + i as usize] as u32;
            let pred = ((f + b + 1) >> 1) as i32;
            sum += (c - pred).unsigned_abs();
        }
    }
    sum
}

/// Encode an intra MB into the bitstream, discarding the reconstruction.
/// Used for intra-fallback MBs in B-pictures (since B-frames are never
/// reference pictures, we don't need to keep the reconstruction).
fn encode_mb_intra_throwaway(
    bw: &mut BitWriter,
    enc: &Mpeg1VideoEncoder,
    v: &VideoFrame,
    mb_row: usize,
    mb_col: usize,
    dc_pred_q: &mut [i32; 3],
) -> Result<()> {
    let mb_w = (enc.width as usize).div_ceil(16);
    let mb_h = (enc.height as usize).div_ceil(16);
    let y_stride = mb_w * 16;
    let c_stride = mb_w * 8;
    let mut recon_y = vec![0u8; y_stride * mb_h * 16];
    let mut recon_cb = vec![0u8; c_stride * mb_h * 8];
    let mut recon_cr = vec![0u8; c_stride * mb_h * 8];
    encode_mb_intra(
        bw,
        enc,
        v,
        mb_row,
        mb_col,
        dc_pred_q,
        &mut recon_y,
        &mut recon_cb,
        &mut recon_cr,
        y_stride,
        c_stride,
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::dct_dc;
    use crate::vlc;
    use oxideav_core::bits::BitReader;

    #[test]
    fn signed_field_round_trip_dc() {
        for size in 1u32..=8 {
            let max_at_size = (1i32 << size) - 1;
            let min_pos = 1i32 << (size - 1);
            let mut values: Vec<i32> = (min_pos..=max_at_size).collect();
            values.extend((min_pos..=max_at_size).map(|v| -v));
            for v in values {
                let bits = encode_signed_field(v, size);
                let vt = 1u32 << (size - 1);
                let decoded = if bits < vt {
                    (bits as i32) - ((1i32 << size) - 1)
                } else {
                    bits as i32
                };
                assert_eq!(decoded, v, "size={size} value={v} bits={bits:b}");
            }
        }
    }

    #[test]
    fn dc_size_lookup_round_trip() {
        let luma = dct_dc::luma();
        for size in 0u8..=8 {
            let entry = lookup_value(luma, size).unwrap_or_else(|| panic!("no entry for {size}"));
            let mut bw = BitWriter::new();
            bw.write_bits(entry.code, entry.bits as u32);
            bw.align_to_byte();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let decoded = vlc::decode(&mut br, luma).expect("decode dc size");
            assert_eq!(decoded, size);
        }
    }

    #[test]
    fn motion_code_round_trip_via_vlc() {
        // Each |motion_code| ∈ 0..=16 must round-trip through the encoder
        // entry → decoder VLC.
        let tbl = mv_tbl::table();
        for abs in 0u8..=16 {
            let e = lookup_motion_code(abs).expect("encode entry");
            let mut bw = BitWriter::new();
            bw.write_bits(e.code, e.bits as u32);
            bw.align_to_byte();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let got = vlc::decode(&mut br, tbl).expect("decode motion code");
            assert_eq!(got, abs);
        }
    }

    #[test]
    fn mv_diff_round_trip_zero_predictor() {
        // For each candidate target half-pel mv with |mv|<=14 (integer pel
        // ±7), encoding from predictor=0 and decoding via the spec rules
        // must reproduce the target.
        for target in (-14..=14).filter(|t| t % 2 == 0) {
            let mut bw = BitWriter::new();
            let mut pred = 0i32;
            encode_one_mv_component(&mut bw, &mut pred, target).expect("encode mv");
            assert_eq!(pred, target, "encoder predictor mismatch");
            bw.align_to_byte();
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            let mut dpred = 0i32;
            let got = crate::motion::decode_motion_component(&mut br, 1, false, &mut dpred)
                .expect("decode mv");
            assert_eq!(got, target, "decoded mv mismatch");
            assert_eq!(dpred, target, "decoder predictor mismatch");
        }
    }
}
