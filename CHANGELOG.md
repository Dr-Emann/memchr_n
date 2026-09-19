# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5](https://github.com/Dr-Emann/memchr_n/compare/v0.1.4...v0.1.5) - 2026-09-19

### Performance
- Faster counting on sse2 + sse4.2 (by @Dr-Emann) - #12

## [0.1.4](https://github.com/Dr-Emann/memchr_n/compare/v0.1.3...v0.1.4) - 2026-09-18

### Added
- Introduce a ByteSet which can build a set of bytes incrementally, and at const time (by @Dr-Emann) - #10

## [0.1.3](https://github.com/Dr-Emann/memchr_n/compare/v0.1.2...v0.1.3) - 2026-09-18

### Added
- Use grouped nibble lookups for larger byte sets ([#7](https://github.com/Dr-Emann/memchr_n/pull/7)) (by @Dr-Emann) - #7

### Other
- Eliminate bounds checks when indexing iterator results ([#8](https://github.com/Dr-Emann/memchr_n/pull/8)) (by @Dr-Emann) - #8

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
