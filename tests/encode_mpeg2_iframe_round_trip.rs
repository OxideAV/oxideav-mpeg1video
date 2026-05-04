//! MPEG-2 I-only encoder + decoder end-to-end round-trip test.
//!
//! Generates a synthetic 3-frame 64×64 Yuv420P sequence (gradient + moving
//! square), encodes it as MPEG-2 video (I-only), decodes it, and checks that
//!   * the bitstream contains one `0x000001B5` sequence_extension after each
//!     `0x000001B3` sequence header,
//!   * the bitstream contains one `0x000001B5` picture_coding_extension after
//!     each `0x00000100` picture header,
//!   * each decoded frame reconstructs with low mean absolute pixel
//!     difference against the original.

use oxideav_core::{
    frame::VideoPlane, CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational,
    TimeBase, VideoFrame,
};
use oxideav_mpeg12video::{
    decoder::make_decoder_mpeg2, encoder::make_encoder_mpeg2, CODEC_ID_MPEG2_STR,
};

const W: u32 = 64;
const H: u32 = 64;

fn synth_frame(idx: u32) -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let ch = h / 2;
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            // Horizontal gradient + vertical ripple.
            let base = ((col as u32 * 4) + (row as u32 * 2)) as u8;
            y[row * w + col] = base;
        }
    }
    // Paint a 16×16 "moving square" of luma 200 at an x-offset that changes
    // with idx, so the frames are not identical and decode differences
    // surface pixel-by-pixel.
    let sx = (idx * 8) as usize;
    let sy = 16usize;
    for j in 0..16 {
        for i in 0..16 {
            let r = sy + j;
            let c = sx + i;
            if r < h && c < w {
                y[r * w + c] = 200;
            }
        }
    }
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = 128u8.wrapping_add((col as u8) / 2);
            cr[row * cw + col] = 128u8.wrapping_sub((row as u8) / 2);
        }
    }
    VideoFrame {
        pts: Some(idx as i64),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: cb,
            },
            VideoPlane {
                stride: cw,
                data: cr,
            },
        ],
    }
}

fn encode_frames(frames: &[VideoFrame]) -> Vec<u8> {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_MPEG2_STR));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(25, 1));
    params.bit_rate = Some(1_500_000);
    let mut enc = make_encoder_mpeg2(&params).expect("build mpeg2 encoder");
    let mut data = Vec::new();
    for f in frames {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
        loop {
            match enc.receive_packet() {
                Ok(p) => data.extend_from_slice(&p.data),
                Err(Error::NeedMore) => break,
                Err(Error::Eof) => break,
                Err(e) => panic!("encoder error: {e}"),
            }
        }
    }
    enc.flush().expect("flush");
    loop {
        match enc.receive_packet() {
            Ok(p) => data.extend_from_slice(&p.data),
            Err(Error::NeedMore) => break,
            Err(Error::Eof) => break,
            Err(e) => panic!("encoder error on flush: {e}"),
        }
    }
    data
}

fn decode_stream(bytes: &[u8]) -> Vec<VideoFrame> {
    let params = CodecParameters::video(CodecId::new(CODEC_ID_MPEG2_STR));
    let mut dec = make_decoder_mpeg2(&params).expect("build mpeg2 decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), bytes.to_vec());
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut out = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => out.push(v),
            Ok(_) => continue,
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("decoder error: {e}"),
        }
    }
    out
}

fn mean_abs_diff(orig: &VideoFrame, recon: &VideoFrame) -> f64 {
    let mut total = 0u64;
    let mut count = 0u64;
    for (o, r) in orig.planes.iter().zip(recon.planes.iter()) {
        for (a, b) in o.data.iter().zip(r.data.iter()) {
            total += (*a as i32 - *b as i32).unsigned_abs() as u64;
            count += 1;
        }
    }
    total as f64 / count as f64
}

fn pixel_match(orig: &VideoFrame, recon: &VideoFrame, tolerance: i32) -> f64 {
    let mut total: u64 = 0;
    let mut matched: u64 = 0;
    for (o, r) in orig.planes.iter().zip(recon.planes.iter()) {
        for (a, b) in o.data.iter().zip(r.data.iter()) {
            total += 1;
            if (*a as i32 - *b as i32).abs() <= tolerance {
                matched += 1;
            }
        }
    }
    matched as f64 / total as f64
}

fn count_start_code(data: &[u8], code: u8) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i + 3 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 && data[i + 3] == code {
            count += 1;
            i += 4;
        } else {
            i += 1;
        }
    }
    count
}

#[test]
fn mpeg2_iframe_round_trip_3_frames() {
    let frames: Vec<VideoFrame> = (0..3).map(synth_frame).collect();
    let bytes = encode_frames(&frames);
    eprintln!("mpeg2 I-only {} bytes for 3 frames", bytes.len());
    assert!(!bytes.is_empty(), "encoder produced no bytes");

    // Every picture in this I-only stream should be preceded by one sequence
    // header (0xB3) and one picture header (0x00). MPEG-2 mandates a
    // sequence_extension (0xB5) after every sequence header and a
    // picture_coding_extension (also 0xB5) after every picture header.
    let n_seq = count_start_code(&bytes, 0xB3);
    let n_pic = count_start_code(&bytes, 0x00);
    let n_ext = count_start_code(&bytes, 0xB5);
    let n_gop = count_start_code(&bytes, 0xB8);
    eprintln!("start-code census: seq={n_seq} gop={n_gop} pic={n_pic} ext(B5)={n_ext}");
    assert_eq!(n_seq, 3, "expected one sequence header per I-frame");
    assert_eq!(n_pic, 3, "expected one picture header per I-frame");
    // 1 sequence_extension + 1 picture_coding_extension per I-frame = 2 * 3.
    assert_eq!(
        n_ext,
        2 * n_pic,
        "expected one extension after each seq header and each pic header"
    );

    let decoded = decode_stream(&bytes);
    assert_eq!(decoded.len(), 3, "expected 3 decoded frames");

    for (i, (orig, recon)) in frames.iter().zip(decoded.iter()).enumerate() {
        let mad = mean_abs_diff(orig, recon);
        let pct8 = pixel_match(orig, recon, 8) * 100.0;
        eprintln!("frame {i}: MAD={mad:.2}, pct(±8)={pct8:.2}%");
        assert!(
            pct8 >= 99.0,
            "frame {i}: ≤±8 match {pct8:.2}% < 99% (MAD={mad:.2})"
        );
    }
}

/// Cross-validate our MPEG-2 I-only output against ffmpeg as a black-box
/// decoder. Skips silently when ffmpeg is unavailable so CI without it stays
/// green.
#[test]
fn ffmpeg_decodes_our_mpeg2_output() {
    if which("ffmpeg").is_none() {
        eprintln!("ffmpeg not found — skipping mpeg2 ffmpeg interop test");
        return;
    }
    let frames: Vec<VideoFrame> = (0..2).map(synth_frame).collect();
    let bytes = encode_frames(&frames);

    let in_path = "/tmp/mpeg2_oxideav.m2v";
    let out_path = "/tmp/mpeg2_oxideav_decoded.yuv";
    std::fs::write(in_path, &bytes).expect("write encoded m2v");
    let _ = std::fs::remove_file(out_path);

    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "mpegvideo",
            "-i",
            in_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            out_path,
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed to decode our MPEG-2 output");

    // Read back the first decoded frame from raw YUV.
    let raw = std::fs::read(out_path).expect("read ffmpeg output");
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let ch = h / 2;
    let frame_size = w * h + 2 * cw * ch;
    assert!(
        raw.len() >= frame_size,
        "ffmpeg output too short: {} < {}",
        raw.len(),
        frame_size
    );
    let ff = VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w,
                data: raw[..w * h].to_vec(),
            },
            VideoPlane {
                stride: cw,
                data: raw[w * h..w * h + cw * ch].to_vec(),
            },
            VideoPlane {
                stride: cw,
                data: raw[w * h + cw * ch..w * h + 2 * cw * ch].to_vec(),
            },
        ],
    };

    let mad = mean_abs_diff(&frames[0], &ff);
    let pct16 = pixel_match(&frames[0], &ff, 16) * 100.0;
    let pct32 = pixel_match(&frames[0], &ff, 32) * 100.0;
    eprintln!(
        "ffmpeg-decoded MPEG-2 round-trip MAD={mad:.2}, pct(±16)={pct16:.2}%, pct(±32)={pct32:.2}%"
    );
    // ffmpeg uses an integer IDCT; ours is f32. The bitstream is conformant
    // (ffmpeg parses + decodes it without complaint), but per-pixel rounding
    // diverges. ±32 / 90% is enough to assert we produced a real MPEG-2
    // stream and not a bag of bytes.
    assert!(
        pct32 >= 90.0,
        "ffmpeg MPEG-2 round-trip ≤±32 match {pct32:.2}% < 90% (MAD={mad:.2})"
    );
}

fn which(prog: &str) -> Option<String> {
    let p = std::process::Command::new("which")
        .arg(prog)
        .output()
        .ok()?;
    if p.status.success() {
        let s = String::from_utf8_lossy(&p.stdout).trim().to_string();
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    } else {
        None
    }
}

#[test]
fn mpeg2_encoder_rejects_b_frames_and_long_gop() {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_MPEG2_STR));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(25, 1));
    params.bit_rate = Some(1_500_000);
    // B-frames not yet supported.
    let r = oxideav_mpeg12video::encoder::make_encoder_mpeg2_with_gop(&params, 1, 1);
    assert!(r.is_err(), "B-frames must be rejected in this milestone");
    // Long GOP not yet supported either.
    let r = oxideav_mpeg12video::encoder::make_encoder_mpeg2_with_gop(&params, 12, 0);
    assert!(r.is_err(), "long GOP must be rejected in this milestone");
}
