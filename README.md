# oxideav-mpeg1video

Pure-Rust MPEG-1 Video (ISO/IEC 11172-2) decoder and encoder for oxideav.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework — a
100% pure Rust media transcoding and streaming stack. No C libraries, no FFI
wrappers, no `*-sys` crates.

## Status

* Decoder: I, P and B pictures — forward + backward MC (half-pel bilinear),
  interpolated (bidirectionally-averaged) prediction, intra DCT blocks, and
  display-order reorder driven by `temporal_reference`.
* Encoder: I, P and B pictures. The B-frame encoder adds a display-order
  reorder buffer, per-MB decision between forward / backward / interpolated
  motion-compensated prediction and an intra fallback. GOP layout is
  configurable via `encoder::make_encoder_with_gop(params, gop_size,
  num_b_frames)` — e.g. `num_b_frames = 2` + `gop_size = 9` gives the
  classic `IBBPBBPBB` pattern. The default is `IPP` (no B-frames).

## Usage

```toml
[dependencies]
oxideav-mpeg1video = "0.0"
```

## License

MIT — see [LICENSE](LICENSE).
