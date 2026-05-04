# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- MPEG-2 decoder: honour `alternate_scan` per picture (H.262 §7.3 Figure 7-3
  / `scan[1][]`). Streams with `picture_coding_extension.alternate_scan = 1`
  are now decoded instead of returning `Error::Unsupported`.
- MPEG-2 decoder: honour `q_scale_type = 1` per picture, mapping the 5-bit
  `quantiser_scale_code` through H.262 §7.4.2.2 Table 7-6 (range up to 112).
  Both the slice-header code and per-MB `quantiser_scale_code` overrides are
  routed through the new lookup. Previously rejected with
  `Error::Unsupported`.
- Test: `ffmpeg_decodes_our_mpeg2_output` cross-validates our MPEG-2 I-only
  encoder against ffmpeg as a black-box decoder (skips silently when ffmpeg
  is unavailable).

## [0.0.9](https://github.com/OxideAV/oxideav-mpeg12video/compare/v0.0.8...v0.0.9) - 2026-05-03

### Other

- cargo fmt: pending rustfmt cleanup
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- adopt slim VideoFrame shape
- pin release-plz to patch-only bumps

## [0.0.8](https://github.com/OxideAV/oxideav-mpeg12video/compare/v0.0.7...v0.0.8) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core

## [0.0.7](https://github.com/OxideAV/oxideav-mpeg12video/compare/v0.0.6...v0.0.7) - 2026-04-24

### Other

- bump criterion 0.5 → 0.8

## [0.0.6](https://github.com/OxideAV/oxideav-mpeg12video/compare/v0.0.5...v0.0.6) - 2026-04-19

### Other

- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- claim AVI FourCCs via oxideav-codec CodecTag registry
- bump oxideav-core to 0.0.5
- migrate to oxideav_core::bits shared BitReader / BitWriter

## [0.0.5](https://github.com/OxideAV/oxideav-mpeg12video/compare/v0.0.4...v0.0.5) - 2026-04-18

### Other

- document decoder-path optimizations + bench numbers
- add MC unclipped fast path (~-6% MPEG-1 inter decode)
- release v0.0.3

## [0.0.4](https://github.com/OxideAV/oxideav-mpeg12video/releases/tag/v0.0.4) - 2026-04-18

### Other

- bump version to 0.0.4
- satisfy cargo fmt + clippy
- add MPEG-2 (H.262) support, rename crate, optimize VLC
- update README + crate description to reflect I/P/B encoder
- add B-frame encoder (FWD/BWD/BI + reorder buffer)
- make crate standalone (pin deps, add CI + release-plz + LICENSE)
- add Decoder::reset overrides for video decoders
- move repo to OxideAV/oxideav-workspace
- add publish metadata (readme/homepage/keywords/categories)
- complete P-frame encode with half-pel ME refinement
- add P-frame encoder (forward MC + residual)
- add I-frame encoder
- full I+P+B frame decode with display-order reordering
- fix I-frame decode — always read EOB after intra AC loop
- scaffold MPEG-1 video decoder (ISO/IEC 11172-2) — headers + VLC tables
