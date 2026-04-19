# oxideav-mpeg12video

Pure-Rust **MPEG-1 Video** (ISO/IEC 11172-2) and **MPEG-2 Video** (H.262 /
ISO/IEC 13818-2) decoder and encoder — I / P / B pictures, forward +
backward half-pel motion compensation, interpolated (bidirectionally-
averaged) prediction, 4:2:0 chroma, and a display-order reorder buffer on
the encoder side. Zero C dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-mpeg12video = "0.0"
```

## Decoder

Feed packets from any demuxer (or a raw elementary stream) and pull
`VideoFrame`s back. Output pixel format is always `Yuv420P`. Use codec id
`"mpeg1video"` for MPEG-1 bitstreams and `"mpeg2video"` for MPEG-2 — a
single `register()` call wires both up.

```rust
use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

let mut codecs = CodecRegistry::new();
oxideav_mpeg12video::register(&mut codecs);

// MPEG-1:
let params = CodecParameters::video(CodecId::new("mpeg1video"));
// MPEG-2:
// let params = CodecParameters::video(CodecId::new("mpeg2video"));
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
- MPEG-2 `sequence_extension` and `picture_coding_extension` parsing
  (progressive Main Profile @ Main Level 4:2:0). Unsupported features
  (interlaced, 4:2:2/4:4:4, `alternate_scan`, `intra_vlc_format=1`,
  non-linear `q_scale_type`, field / dual-prime / 16×8 MVs, concealment
  MVs, scalable extensions) are rejected with a clear `Error::Unsupported`
  rather than mis-decoded.
- Picture types I, P, B (D-pictures are rejected — obsolete, MPEG-1 only).
- Forward + backward motion compensation with half-pel bilinear
  interpolation; interpolated (averaged) B-frame prediction.
- Skipped-MB motion-vector inheritance (previous-MB MV for P, last fwd/bwd
  for B).
- Intra / non-intra DCT block decode, custom quantiser matrices from the
  sequence header (and from MPEG-2 `quant_matrix_extension`). MPEG-2
  dequantisation + global-XOR mismatch + 12-bit signed escape form.
- Display-order reordering driven by `temporal_reference` + the
  most-recent GOP anchor. B-pictures are emitted in place; I/P pictures
  are emitted one anchor late so trailing B-pictures can reference them.
- 4:2:0 chroma, 12-bit max dimensions (4095 × 4095 for MPEG-1; MP@ML
  1920 × 1152 capability hint for MPEG-2).

## Encoder

The encoder accepts frames in display order and emits an MPEG-1 or MPEG-2
elementary stream (sequence header, GOP header, picture headers, one slice
per MB row). The MPEG-1 default GOP is `IPP` — no B-frames — to keep
cumulative drift in the f32 IDCT chain bounded on long sequences. Use
`make_encoder_with_gop` for arbitrary `I B*N P B*N P ...` layouts.

The MPEG-2 encoder currently produces **I-only** bitstreams (progressive
Main Profile @ Main Level 4:2:0). MPEG-2 P/B encoding is a later
milestone.

```rust
use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, Rational};
use oxideav_mpeg12video::encoder::make_encoder_with_gop;

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

For MPEG-2 I-only encoding, use `make_encoder_mpeg2` with codec id
`"mpeg2video"`.

Encoder coverage:

- Sequence + GOP + picture header write, one slice per macroblock row.
- Input frames must be `Yuv420P` at the configured size; other formats
  are rejected with `Error::unsupported`.
- Frame rates: MPEG-1 Table 2-D.4 codes 1 through 8 (23.976 / 24 / 25 /
  29.97 / 30 / 50 / 59.94 / 60). MPEG-2 reuses the same table.
- I-pictures: forward DCT, intra quantisation, DC differential + AC
  run/level VLC. MPEG-2 emits a `sequence_extension` after each sequence
  header and a `picture_coding_extension` after each picture header; uses
  the MPEG-2 intra dequant formula (`(level * q * W) / 16`) and global
  mismatch, and writes escape run/level pairs as `run(6) + signed 12-bit
  level`.
- P-pictures (MPEG-1 only): integer-pel block-matching motion estimation
  at ±8 with half-pel refinement, MV differential via Table B-10, MB types
  skip / forward / forward+pattern / intra fallback, CBP via Table B-9,
  non-intra quant + Table B-14 VLC.
- B-pictures (MPEG-1 only): per-MB decision between forward / backward /
  interpolated (fwd + bwd averaged) MC and intra fallback; display-order
  reorder buffer emits each anchor before its preceding B-pictures.
  `forward_f_code` and `backward_f_code` are both 1 (±16 half-pel MV
  range).
- Reconstructed references are kept in-encoder so decoder output is
  drift-free with respect to what the encoder predicted from.
- Closed-GOP semantics: B-frames that would straddle a GOP boundary are
  promoted to P-frames.

## Codec IDs

`"mpeg1video"` (MPEG-1 Video, ISO/IEC 11172-2). `"mpeg2video"` (MPEG-2
Video / H.262, ISO/IEC 13818-2). Accepted / produced pixel format for both
is `Yuv420P`.

## Performance

The decoder pipeline has two hot paths that got focused attention:

- **VLC decode** — every table (AC coefficients, DC size, motion code,
  MBA, CBP, MB type) is wrapped in a [`VlcTable`] that pre-computes the
  longest codeword and builds a 512-entry 9-bit prefix LUT at first use.
  Any codeword ≤ 9 bits is resolved in a single array lookup; only the
  rare longer codes fall through to the linear scan. Because the hot AC
  coefficient table has 113 entries with codewords up to 17 bits, the
  caching + LUT together cut AC decode cost dramatically on typical
  content.
- **Motion compensation** — `mc_block` checks whether the source window
  (including the half-pel partner sample) sits fully inside the reference
  plane. Interior MBs take an unclipped fast path that uses
  `slice::copy_from_slice` for the integer case and works on sliced
  reference rows for the half-pel cases. Border MBs still take the
  original clamped path.

Measured on release builds with `cargo bench -p oxideav-mpeg12video`
(256×256 synthetic content, 3 frames for the MPEG-1 IPPP bench):

| Bench                                | Before  | After   | Δ      |
|--------------------------------------|---------|---------|--------|
| MPEG-2 I-frame decode                | 400 µs  | 241 µs  | −40 %  |
| MPEG-1 IPPP 3-frame decode           | 801 µs  | 752 µs  | −6.3 % |
| MPEG-2 I-frame encode                | 570 µs  | 574 µs  | ~flat  |
| IDCT, 1000 8×8 blocks                | 51.5 µs | 51.4 µs | flat   |

The IDCT kernel remains the textbook separable O(N²) cosine-table
multiply. A fast separable IDCT (LLM / Chen-Wang) is a likely next
speedup target for the decoder.

Benchmarks live in `benches/dct_bench.rs` (IDCT / FDCT micro-benches) and
`benches/frame_bench.rs` (full encode + decode round-trips).

## License

MIT - see [LICENSE](LICENSE).
