# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.8](https://github.com/Dr-Emann/memchr_n/compare/v0.1.7...v0.1.8) - 2026-09-27

### Fixed
- Omit haystack bytes from iterator Debug and mark unused results (by @Dr-Emann) - #30
- Correct changelog links and deduplicate PR references (by @Dr-Emann) - #27
- Benchmark the vector bitset kernel (by @Dr-Emann) - #26 [#7](https://github.com/Dr-Emann/memchr_n/issues/7)

### Other
- Remove Cargo.lock from fuzzer directory (by @Dr-Emann)
- Fuzz mixed iterator operations against scalar model (by @Dr-Emann) - #32
- Share benchmark cases and shuffle eligibility (by @Dr-Emann) - #31
- Cover explicit SIMD levels and lint ARM builds (by @Dr-Emann) - #29
- Complete publication metadata and licenses (by @Dr-Emann) - #28
- Clarify {add/remove}_range behavior with exhausted ranges (by @Dr-Emann) - #25
- Add section explaining the (minimal) use of `unsafe` in the crate (by @Dr-Emann) - #22

## [0.1.7](https://github.com/Dr-Emann/memchr_n/compare/v0.1.6...v0.1.7) - 2026-09-22

### Other
- Update to fearless_simd 1.0 (by @Dr-Emann) - [#20](https://github.com/Dr-Emann/memchr_n/pull/20)

## [0.1.6](https://github.com/Dr-Emann/memchr_n/compare/v0.1.5...v0.1.6) - 2026-09-19

### Other
- Update readme performance section ([#16](https://github.com/Dr-Emann/memchr_n/pull/16)) (by @Dr-Emann)
- *(deps)* Update to fearless-simd 1.0.0-rc.2 ([#17](https://github.com/Dr-Emann/memchr_n/pull/17)) (by @Dr-Emann)

### Performance
- Preserve SIMD scan arguments in registers ([#19](https://github.com/Dr-Emann/memchr_n/pull/19)) (by @Dr-Emann)

## [0.1.5](https://github.com/Dr-Emann/memchr_n/compare/v0.1.4...v0.1.5) - 2026-09-19

### Performance
- Faster counting on sse2 + sse4.2 (by @Dr-Emann) - [#12](https://github.com/Dr-Emann/memchr_n/pull/12)

## [0.1.4](https://github.com/Dr-Emann/memchr_n/compare/v0.1.3...v0.1.4) - 2026-09-18

### Added
- Introduce a ByteSet which can build a set of bytes incrementally, and at const time (by @Dr-Emann) - [#10](https://github.com/Dr-Emann/memchr_n/pull/10)

## [0.1.3](https://github.com/Dr-Emann/memchr_n/compare/v0.1.2...v0.1.3) - 2026-09-18

### Added
- Use grouped nibble lookups for larger byte sets ([#7](https://github.com/Dr-Emann/memchr_n/pull/7)) (by @Dr-Emann)

### Other
- Eliminate bounds checks when indexing iterator results ([#8](https://github.com/Dr-Emann/memchr_n/pull/8)) (by @Dr-Emann)

## [0.1.2](https://github.com/Dr-Emann/memchr_n/compare/v0.1.1...v0.1.2) - 2026-09-17

### Added
- Add `Iter::advance_to` to advance to an index in the haystack (by @Dr-Emann)

## [0.1.1](https://github.com/Dr-Emann/memchr_n/compare/v0.1.0...v0.1.1) - 2026-09-17

### Added
- Add Clone, Debug, and FusedIterator for Iter (by @Dr-Emann)

### Fixed
- Select SWAR automatically when SIMD is unavailable (by @Dr-Emann)
- Preserve byte order when staging short vector inputs (by @Dr-Emann)
- Preserve byte order in SWAR bitset lookup (by @Dr-Emann)

### Other
- Centralize the SearchPlan invariant (by @Dr-Emann)
- Lower MSRV to Rust 1.89 (by @Dr-Emann)
