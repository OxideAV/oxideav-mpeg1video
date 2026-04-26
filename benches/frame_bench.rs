//! End-to-end benchmarks: encode one MPEG-2 I-frame, decode one MPEG-2
//! I-frame. These exercise the full pipeline (header parsing, slice /
//! macroblock decode, block-level VLC + dequant + IDCT, motion
//! compensation paths).
//!
//! We use MPEG-2 I-only because it's the encoder we ship, but the decoder
//! hot paths are almost identical for MPEG-1 decoding as well — both
//! dispatch through the same `block.rs` / `mb.rs` code gated by the
//! `PictureParams` passed in.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use oxideav_core::{
    frame::VideoPlane, CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational,
    TimeBase, VideoFrame,
};
use oxideav_mpeg12video::{
    decoder::{make_decoder, make_decoder_mpeg2},
    encoder::{make_encoder, make_encoder_mpeg2},
    CODEC_ID_MPEG2_STR, CODEC_ID_STR,
};

const W: u32 = 256;
const H: u32 = 256;

fn synth_frame() -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let ch = h / 2;
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            // Gradient + ripple so there's real AC content for the encoder.
            let v = ((col as u32 * 3) + (row as u32 * 2)) as i32 + ((col as i32 ^ row as i32) & 31);
            y[row * w + col] = (v.clamp(0, 255)) as u8;
        }
    }
    let mut cb = vec![128u8; cw * ch];
    let mut cr = vec![128u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = 128u8.wrapping_add((col as u8) & 31);
            cr[row * cw + col] = 128u8.wrapping_sub((row as u8) & 31);
        }
    }
    VideoFrame {
        pts: Some(0),
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

fn encode_params() -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_MPEG2_STR));
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p.frame_rate = Some(Rational::new(25, 1));
    p.bit_rate = Some(4_000_000);
    p
}

fn encode_one_frame(frame: &VideoFrame) -> Vec<u8> {
    let params = encode_params();
    let mut enc = make_encoder_mpeg2(&params).expect("encoder");
    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    enc.flush().expect("flush");
    let mut data = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => data.extend_from_slice(&p.data),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("enc: {e}"),
        }
    }
    data
}

fn bench_encode(c: &mut Criterion) {
    let frame = synth_frame();
    let mut group = c.benchmark_group("encode");
    group.throughput(Throughput::Elements(1));
    group.bench_function("mpeg2_i_256x256", |b| {
        b.iter(|| encode_one_frame(&frame));
    });
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let frame = synth_frame();
    let bytes = encode_one_frame(&frame);
    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Elements(1));
    group.bench_function("mpeg2_i_256x256", |b| {
        b.iter(|| {
            let params = CodecParameters::video(CodecId::new(CODEC_ID_MPEG2_STR));
            let mut dec = make_decoder_mpeg2(&params).expect("decoder");
            let pkt = Packet::new(0, TimeBase::new(1, 25), bytes.clone());
            dec.send_packet(&pkt).expect("send");
            dec.flush().expect("flush");
            let mut frames = 0;
            loop {
                match dec.receive_frame() {
                    Ok(_) => frames += 1,
                    Err(Error::NeedMore) | Err(Error::Eof) => break,
                    Err(e) => panic!("dec: {e}"),
                }
            }
            frames
        });
    });
    group.finish();
}

fn mpeg1_params() -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p.frame_rate = Some(Rational::new(25, 1));
    p.bit_rate = Some(4_000_000);
    p
}

fn encode_mpeg1_gop(frames: &[VideoFrame]) -> Vec<u8> {
    let params = mpeg1_params();
    let mut enc = make_encoder(&params).expect("mpeg1 encoder");
    let mut data = Vec::new();
    for f in frames {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        loop {
            match enc.receive_packet() {
                Ok(p) => data.extend_from_slice(&p.data),
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(e) => panic!("enc: {e}"),
            }
        }
    }
    enc.flush().expect("flush");
    loop {
        match enc.receive_packet() {
            Ok(p) => data.extend_from_slice(&p.data),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("enc: {e}"),
        }
    }
    data
}

fn bench_decode_mpeg1_ippp(c: &mut Criterion) {
    // A 3-frame IPP GOP exercises inter decode (motion compensation) in
    // addition to intra decode.
    let frames: Vec<VideoFrame> = (0..3)
        .map(|i| {
            let mut f = synth_frame();
            // Nudge each frame so motion estimation has something to work on.
            let shift = (i * 2) as usize;
            let w = W as usize;
            let h = H as usize;
            for row in 0..h {
                for col in 0..w {
                    let orig = f.planes[0].data[row * w + col];
                    f.planes[0].data[row * w + col] = orig.wrapping_add(shift as u8);
                }
            }
            f
        })
        .collect();
    let bytes = encode_mpeg1_gop(&frames);
    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Elements(3));
    group.bench_function("mpeg1_ippp_256x256", |b| {
        b.iter(|| {
            let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
            let mut dec = make_decoder(&params).expect("decoder");
            let pkt = Packet::new(0, TimeBase::new(1, 25), bytes.clone());
            dec.send_packet(&pkt).expect("send");
            dec.flush().expect("flush");
            let mut frames = 0;
            loop {
                match dec.receive_frame() {
                    Ok(_) => frames += 1,
                    Err(Error::NeedMore) | Err(Error::Eof) => break,
                    Err(e) => panic!("dec: {e}"),
                }
            }
            frames
        });
    });
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode, bench_decode_mpeg1_ippp);
criterion_main!(benches);
