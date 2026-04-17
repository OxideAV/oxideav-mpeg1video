# oxideav-mpeg1video

Pure-Rust **MPEG-1 Video** (ISO/IEC 11172-2) decoder and encoder — I / P / B
pictures, forward + backward half-pel motion compensation, interpolated
(bidirectionally-averaged) prediction, 4:2:0 chroma, and a display-order
reorder buffer on the encoder side. Zero C dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.0"
oxideav-codec = "0.0"
oxideav-mpeg1video = "0.0"
```

## Decoder

Feed packets from any demuxer (or a raw elementary stream) and pull
`VideoFrame`s back. Output pixel format is always `Yuv420P`.

```rust
use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

let mut codecs = CodecRegistry::new();
oxideav_mpeg1video::register(&mut codecs);

let params = CodecParameters::video(CodecId::new("mpeg1video"));
let mut dec = codecs.make_decoder(&params)?;

let pkt = Packet::new(0, TimeBase::new(1, 24), es_bytes).with_pts(0);
dec.send_packet(&pkt)?;
dec.flush()?;
loop {
    match dec.receive_frame() {
        Ok(Frame::Video(vf)) => {
            // vf.format == PixelFormat::Yuv420P
            // vf.planes = [Y, Cb, Cr]; vf.pts is in display order.
        }
        Ok(_) => continue,
        Err(oxideav_core::Error::Eof) | Err(oxideav_core::Error::NeedMore) => break,
        Err(e) => return Err(e.into()),
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Decoder coverage:

- Sequence header, GOP header, picture header, slice / macroblock / block
  layers.
- Picture types I, P, B (D-pictures are rejected — obsolete, MPEG-1 only).
- Forward + backward motion compensation with half-pel bilinear
  interpolation; interpolated (averaged) B-frame prediction.
- Skipped-MB motion-vector inheritance (previous-MB MV for P, last fwd/bwd
  for B).
- Intra / non-intra DCT block decode, custom quantiser matrices from the
  sequence header.
- Display-order reordering driven by `temporal_reference` + the
  most-recent GOP anchor. B-pictures are emitted in place; I/P pictures
  are emitted one anchor late so trailing B-pictures can reference them.
- 4:2:0 chroma, 12-bit max dimensions (4095 x 4095).

## Encoder

The encoder accepts frames in display order and emits an MPEG-1 elementary
stream (sequence header, GOP header, picture headers, one slice per MB
row). The default GOP is `IPP` — no B-frames — to keep cumulative drift
in the f32 IDCT chain bounded on long sequences. Use
`make_encoder_with_gop` for arbitrary `I B*N P B*N P ...` layouts.

```rust
use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, Rational};
use oxideav_mpeg1video::encoder::make_encoder_with_gop;

let mut params = CodecParameters::video(CodecId::new("mpeg1video"));
params.width = Some(320);
params.height = Some(240);
params.pixel_format = Some(PixelFormat::Yuv420P);
params.frame_rate = Some(Rational::new(24, 1));
params.bit_rate = Some(1_500_000);

// `IBBPBBPBB` GOP (gop_size = 9, num_b_frames = 2). Use
// `make_encoder` for the default `IPP` layout.
let mut enc = make_encoder_with_gop(&params, 9, 2)?;
for f in display_order_frames {
    enc.send_frame(&Frame::Video(f))?;
}
enc.flush()?;
while let Ok(pkt) = enc.receive_packet() {
    sink.write_all(&pkt.data)?;
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Encoder coverage:

- Sequence + GOP + picture header write, one slice per macroblock row.
- Input frames must be `Yuv420P` at the configured size; other formats
  are rejected with `Error::unsupported`.
- Frame rates: MPEG-1 Table 2-D.4 codes 1 through 8 (23.976 / 24 / 25 /
  29.97 / 30 / 50 / 59.94 / 60).
- I-pictures: forward DCT, intra quantisation, DC differential + AC
  run/level VLC.
- P-pictures: integer-pel block-matching motion estimation at +/- 8 with
  half-pel refinement, MV differential via Table B-10, MB types
  skip / forward / forward+pattern / intra fallback, CBP via Table B-9,
  non-intra quant + Table B-14 VLC.
- B-pictures: per-MB decision between forward / backward / interpolated
  (fwd + bwd averaged) MC and intra fallback; display-order reorder buffer
  emits each anchor before its preceding B-pictures. `forward_f_code` and
  `backward_f_code` are both 1 (+/- 16 half-pel MV range).
- Reconstructed references are kept in-encoder so decoder output is
  drift-free with respect to what the encoder predicted from.
- Closed-GOP semantics: B-frames that would straddle a GOP boundary are
  promoted to P-frames.

## Codec ID

`"mpeg1video"`. Accepted / produced pixel format: `Yuv420P`.

## License

MIT - see [LICENSE](LICENSE).
