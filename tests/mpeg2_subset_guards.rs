//! Decoder subset-guard tests: features explicitly out of scope for the
//! first-pass MPEG-2 milestone should be rejected with `Error::Unsupported`
//! rather than silently producing garbage output.
//!
//! Rather than hand-crafting full MPEG-2 elementary streams (which is tedious
//! and fragile), we use the encoder to generate a known-good stream and then
//! surgically patch the single bit in the picture_coding_extension that
//! triggers each unsupported feature.

use oxideav_core::{
    frame::VideoPlane, CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational,
    TimeBase, VideoFrame,
};
use oxideav_mpeg12video::{
    decoder::make_decoder_mpeg2, encoder::make_encoder_mpeg2, CODEC_ID_MPEG2_STR,
};

fn tiny_frame() -> VideoFrame {
    let w = 32usize;
    let h = 32usize;
    let cw = 16usize;
    let ch = 16usize;
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w,
                data: vec![128u8; w * h],
            },
            VideoPlane {
                stride: cw,
                data: vec![128u8; cw * ch],
            },
            VideoPlane {
                stride: cw,
                data: vec![128u8; cw * ch],
            },
        ],
    }
}

fn encode_one_mpeg2(frame: &VideoFrame) -> Vec<u8> {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_MPEG2_STR));
    params.width = Some(32);
    params.height = Some(32);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(25, 1));
    params.bit_rate = Some(1_000_000);
    let mut enc = make_encoder_mpeg2(&params).expect("build mpeg2 encoder");
    enc.send_frame(&Frame::Video(frame.clone()))
        .expect("send_frame");
    enc.flush().expect("flush");
    let mut data = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => data.extend_from_slice(&p.data),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("encoder error: {e}"),
        }
    }
    data
}

/// Find the byte offset immediately after the second occurrence of
/// `0x000001B5` (sequence_extension is the first, picture_coding_extension
/// the second in an I-only stream).
fn picture_coding_ext_start(data: &[u8]) -> usize {
    let mut hits = 0;
    let mut i = 0;
    while i + 3 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 && data[i + 3] == 0xB5 {
            hits += 1;
            if hits == 2 {
                return i + 4; // first byte of the extension payload
            }
            i += 4;
        } else {
            i += 1;
        }
    }
    panic!("picture_coding_extension not found");
}

fn decode_expect_unsupported(data: &[u8]) -> String {
    let params = CodecParameters::video(CodecId::new(CODEC_ID_MPEG2_STR));
    let mut dec = make_decoder_mpeg2(&params).expect("build mpeg2 decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), data.to_vec());
    // Rejection may arrive on either send_packet (the slice parser triggers
    // the subset guard synchronously while draining the buffer) or
    // receive_frame (if we get to that path).
    match dec.send_packet(&pkt) {
        Err(Error::Unsupported(msg)) => return msg,
        Err(other) => panic!("unexpected send_packet error: {other}"),
        Ok(()) => {}
    }
    let _ = dec.flush();
    match dec.receive_frame() {
        Err(Error::Unsupported(msg)) => msg,
        Ok(_) => panic!("decoder accepted an unsupported subset"),
        Err(Error::NeedMore) | Err(Error::Eof) => {
            panic!("decoder returned no frame and no error")
        }
        Err(other) => panic!("unexpected error variant: {other}"),
    }
}

/// `alternate_scan = 1` is now accepted by the decoder (round-trip coverage
/// for the scan path lives in the block-level unit tests). With a constant
/// 128-grey input every block has only a DC coefficient, so the AC scan
/// order is irrelevant and the bit-flipped stream still reconstructs the
/// same pixels.
#[test]
fn accepts_alternate_scan_for_dc_only_stream() {
    let frame = tiny_frame();
    let mut bytes = encode_one_mpeg2(&frame);
    // picture_coding_extension byte 3 layout (MSB → LSB):
    //   tff fpfdct conceal qst ivlc altscan rff c420
    // Default: 1100_0001 = 0xC1. Set alternate_scan (bit 2): 0xC5.
    let ext_start = picture_coding_ext_start(&bytes);
    assert_eq!(
        bytes[ext_start + 3],
        0xC1,
        "unexpected pic-coding-ext byte3"
    );
    bytes[ext_start + 3] = 0xC5;

    let params = CodecParameters::video(CodecId::new(CODEC_ID_MPEG2_STR));
    let mut dec = make_decoder_mpeg2(&params).expect("build mpeg2 decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), bytes);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let frame = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        other => panic!("expected one VideoFrame, got {other:?}"),
    };
    // All-grey input → all-grey reconstruction regardless of scan order.
    for plane in &frame.planes {
        for &p in &plane.data {
            assert!(
                (p as i32 - 128).abs() <= 2,
                "alternate_scan flipped a DC-only block: pixel {p}"
            );
        }
    }
}

#[test]
fn rejects_intra_vlc_format() {
    let frame = tiny_frame();
    let mut bytes = encode_one_mpeg2(&frame);
    let ext_start = picture_coding_ext_start(&bytes);
    assert_eq!(bytes[ext_start + 3], 0xC1);
    // intra_vlc_format is bit 5 from MSB in byte 3 (bit index 2 from LSB).
    bytes[ext_start + 3] = 0xC5 | 0x04; // set intra_vlc_format=1 + alternate_scan=1 → we want
                                        // Actually let's isolate: set intra_vlc_format=1 only (bit 2 from LSB of byte3).
    bytes[ext_start + 3] = 0xC1 | 0x04; // 0b1100_0101 with bit2 set → 0b1100_0101 = 0xC5
                                        // Wait that hits alternate_scan too. Let me reconsider the bit layout.
                                        //
                                        // byte 3 bits (MSB→LSB): tff, fpfdct, conceal, qst, ivlc, altscan, rff, c420
                                        //                         7    6      5        4    3     2        1    0
                                        // Default: 1100_0001 = 0xC1.
                                        //   bit 3 (= 0x08) is intra_vlc_format
                                        //   bit 2 (= 0x04) is alternate_scan
    bytes[ext_start + 3] = 0xC1 | 0x08; // set intra_vlc_format=1
    let msg = decode_expect_unsupported(&bytes);
    assert!(
        msg.contains("intra_vlc_format"),
        "expected intra_vlc_format rejection, got: {msg}"
    );
}
