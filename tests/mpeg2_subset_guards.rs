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

#[test]
fn rejects_alternate_scan() {
    let frame = tiny_frame();
    let mut bytes = encode_one_mpeg2(&frame);
    // picture_coding_extension layout after the 4-bit extension id (0x8):
    //   4×4 bits  f_code[0..4]           = 2 bytes   (offset bytes 0..2 within payload)
    //   2 bits    intra_dc_precision
    //   2 bits    picture_structure
    //   1 bit     top_field_first
    //   1 bit     frame_pred_frame_dct
    //   1 bit     concealment_motion_vectors
    //   1 bit     q_scale_type
    //   1 bit     intra_vlc_format
    //   1 bit     alternate_scan
    //   1 bit     repeat_first_field
    //   1 bit     chroma_420_type
    //   1 bit     progressive_frame
    //   1 bit     composite_display_flag
    //
    // First 4 bits of byte[0] are the extension ID (0x8), followed by f_code[0][0].
    // Total: 4 (id) + 16 (f_codes) + 2 (dc_prec) + 2 (struct) + 9 flags = 33 bits →
    //   bytes 0..4 cover the full extension, with flags packed into bytes 2-4.
    //
    // Rather than decode bit positions, flip the alternate_scan bit by
    // searching for a specific pattern and XOR-ing it. But simpler: re-write
    // the full ext payload using the bitwriter.
    let ext_start = picture_coding_ext_start(&bytes);
    // payload is 33 bits = 5 bytes (with 7 padding bits). Byte layout:
    //   byte 0: ext_id (4 bits) | f_code[0][0] (4 bits) = 0x8F (f_code=15)
    //   byte 1: f_code[0][1] | f_code[1][0]             = 0xFF
    //   byte 2: f_code[1][1] | intra_dc_prec(2) | pict_struct(2) = 0xFF -> 0b1111_00_11 = 0xF3
    //   byte 3: tff(1)|fpfdct(1)|conceal(1)|qst(1)|ivlc(1)|altscan(1)|rff(1)|c420(1)
    //   byte 4: prog_frame(1)|composite(1)|stuffing(6) -> 0b1000_0000 = 0x80
    //
    // For I-only default: tff=1, fpfdct=1, conceal=0, qst=0, ivlc=0,
    //   altscan=0, rff=0, c420=1 → 0b1100_0001 = 0xC1
    // For the altered variant: altscan=1 → 0b1100_0101 = 0xC5
    assert_eq!(
        bytes[ext_start + 3],
        0xC1,
        "unexpected pic-coding-ext byte3"
    );
    bytes[ext_start + 3] = 0xC5; // set alternate_scan=1
    let msg = decode_expect_unsupported(&bytes);
    assert!(
        msg.contains("alternate_scan"),
        "expected alternate_scan rejection, got: {msg}"
    );
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
