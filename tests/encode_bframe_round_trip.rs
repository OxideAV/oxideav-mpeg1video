//! End-to-end B-frame encoder round-trip test.
//!
//! Verifies:
//!   * The encoder produces a bitstream with explicit B-frame picture type.
//!   * Decoding that bitstream recovers the original frames with PSNR > 25 dB
//!     per plane.
//!   * The bitstream decode-order of picture types differs from the display
//!     order (proving reorder-buffer correctness).

use oxideav_core::{
    frame::VideoPlane, CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational,
    TimeBase, VideoFrame,
};
use oxideav_mpeg12video::{
    decoder::make_decoder,
    encoder::{make_encoder_with_gop, DEFAULT_QUANT_SCALE},
    headers::{parse_picture_header, PictureType},
    start_codes::{self, PICTURE_START_CODE},
    CODEC_ID_STR,
};

fn synth_frame(w: u32, h: u32, t: usize, pts: i64, _tb: TimeBase) -> VideoFrame {
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y = vec![0u8; (w * h) as usize];
    let mut cb = vec![128u8; (cw * ch) as usize];
    let mut cr = vec![128u8; (cw * ch) as usize];
    for j in 0..h as usize {
        for i in 0..w as usize {
            // Diagonal gradient that shifts (+1 px horizontally per frame).
            let ix = i.wrapping_add(t) & 0x3F;
            let jy = j & 0x3F;
            let v = ((ix * 4) ^ (jy * 4)) as u8;
            y[j * w as usize + i] = v;
        }
    }
    for j in 0..ch as usize {
        for i in 0..cw as usize {
            cb[j * cw as usize + i] = 128u8.wrapping_add(((i + t) & 0x1F) as u8);
            cr[j * cw as usize + i] = 128u8.wrapping_add((j & 0x1F) as u8);
        }
    }
    VideoFrame {
        pts: Some(pts),
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: y,
            },
            VideoPlane {
                stride: cw as usize,
                data: cb,
            },
            VideoPlane {
                stride: cw as usize,
                data: cr,
            },
        ],
    }
}

fn encode_bframes(frames: &[VideoFrame], w: u32, h: u32, gop_size: u32, num_b: u32) -> Vec<u8> {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(24, 1));
    params.bit_rate = Some(1_500_000);
    let mut enc = make_encoder_with_gop(&params, gop_size, num_b).expect("build encoder");
    for f in frames {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
    }
    enc.flush().expect("flush");
    let mut data = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => data.extend_from_slice(&p.data),
            Err(Error::NeedMore) => break,
            Err(Error::Eof) => break,
            Err(e) => panic!("encoder error: {e}"),
        }
    }
    data
}

fn decode_frames(bytes: &[u8]) -> Vec<VideoFrame> {
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("build decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 24), bytes.to_vec()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => frames.push(v),
            Ok(_) => continue,
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("decoder error: {e}"),
        }
    }
    frames
}

fn planewise_psnr_db(orig: &VideoFrame, recon: &VideoFrame) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (pi, (o, r)) in orig.planes.iter().zip(recon.planes.iter()).enumerate() {
        let mut sq: f64 = 0.0;
        let mut count: u64 = 0;
        for (a, b) in o.data.iter().zip(r.data.iter()) {
            let d = *a as f64 - *b as f64;
            sq += d * d;
            count += 1;
        }
        out[pi] = if count == 0 {
            f64::INFINITY
        } else {
            let mse = sq / count as f64;
            if mse <= 0.0 {
                f64::INFINITY
            } else {
                10.0 * (255.0_f64 * 255.0 / mse).log10()
            }
        };
    }
    out
}

/// Parse picture types from the bitstream in decode order.
fn picture_types_in_decode_order(bytes: &[u8]) -> Vec<PictureType> {
    let mut out = Vec::new();
    for (pos, code) in start_codes::iter_start_codes(bytes) {
        if code == PICTURE_START_CODE {
            let mut br = oxideav_core::bits::BitReader::new(&bytes[pos + 4..]);
            if let Ok(ph) = parse_picture_header(&mut br) {
                out.push(ph.picture_type);
            }
        }
    }
    out
}

/// Per-task: "encode a 5-frame sequence with GOP `IBPBP`, decode, verify
/// output frames match input by PSNR > 25 dB per plane. Also verify the
/// bitstream frame ORDER matches what a decoder expects (I, P, B, P, B)."
#[test]
fn ibpbp_round_trip_psnr_and_order() {
    // 5-frame sequence with pattern IBPBP (num_b_frames=1, gop_size=5).
    let w = 64u32;
    let h = 64u32;
    let n = 5;
    let tb = TimeBase::new(1, 24);
    let mut frames: Vec<VideoFrame> = Vec::with_capacity(n);
    for t in 0..n {
        frames.push(synth_frame(w, h, t, t as i64, tb));
    }

    let bytes = encode_bframes(&frames, w, h, 5, 1);
    eprintln!("encoded {} bytes for IBPBP 5-frame seq", bytes.len());
    assert!(!bytes.is_empty());
    assert_eq!(&bytes[..4], &[0x00, 0x00, 0x01, 0xB3]);

    // Decode-order picture types — should be I, P, B, P, B (anchor-first
    // emission with B-frames delayed until their backward reference exists).
    let types = picture_types_in_decode_order(&bytes);
    eprintln!("decode-order picture types: {types:?}");
    assert_eq!(
        types,
        vec![
            PictureType::I,
            PictureType::P,
            PictureType::B,
            PictureType::P,
            PictureType::B,
        ],
        "bitstream picture-type order must be I, P, B, P, B"
    );

    // Round-trip decode.
    let recon = decode_frames(&bytes);
    assert_eq!(recon.len(), n, "decoder produced wrong frame count");

    // PSNR per plane per frame, average.
    let mut sum_per_plane = [0.0; 3];
    for (i, (orig, rec)) in frames.iter().zip(recon.iter()).enumerate() {
        let p = planewise_psnr_db(orig, rec);
        eprintln!(
            "IBPBP frame {i}: Y-PSNR={:.2} dB, Cb-PSNR={:.2} dB, Cr-PSNR={:.2} dB",
            p[0], p[1], p[2]
        );
        for k in 0..3 {
            sum_per_plane[k] += p[k];
        }
    }
    let avg_y = sum_per_plane[0] / n as f64;
    let avg_cb = sum_per_plane[1] / n as f64;
    let avg_cr = sum_per_plane[2] / n as f64;
    eprintln!(
        "IBPBP avg PSNR: Y={avg_y:.2} dB, Cb={avg_cb:.2} dB, Cr={avg_cr:.2} dB at qp={}",
        DEFAULT_QUANT_SCALE
    );
    assert!(avg_y > 25.0, "Y-plane avg PSNR {avg_y:.2} ≤ 25 dB");
    assert!(avg_cb > 25.0, "Cb-plane avg PSNR {avg_cb:.2} ≤ 25 dB");
    assert!(avg_cr > 25.0, "Cr-plane avg PSNR {avg_cr:.2} ≤ 25 dB");
}

/// Encode a longer sequence at the default `IBBP` GOP and verify we see
/// some B-frames in the decode-order stream.
#[test]
fn default_gop_emits_b_frames() {
    let w = 64u32;
    let h = 64u32;
    let n = 12;
    let tb = TimeBase::new(1, 24);
    let mut frames: Vec<VideoFrame> = Vec::with_capacity(n);
    for t in 0..n {
        frames.push(synth_frame(w, h, t, t as i64, tb));
    }
    // gop_size=9 with num_b=2 → IBBPBBPBB. 12 frames = one full GOP + IBB of
    // the next GOP (trailing Bs promoted to Ps on flush if no next anchor).
    let bytes = encode_bframes(&frames, w, h, 9, 2);
    let types = picture_types_in_decode_order(&bytes);
    eprintln!("12-frame default IBBP decode-order types: {types:?}");
    let b_count = types.iter().filter(|t| matches!(t, PictureType::B)).count();
    assert!(b_count > 0, "expected at least one B-frame, got {types:?}");
}
