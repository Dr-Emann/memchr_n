# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.2](https://github.com/Dr-Emann/memchr_n/compare/v0.0.1...v0.0.2) - 2026-09-09

### Added

- iterate by chunk pairs, then chunks, then blocks
- add `nth` iterator specialization

### Other

- reorg
- more cleanup
- cleanup
- release-plz setup
- cleanup of comments
- more
- use dispatch
- misc
- update for main fearless_simd
- Simplify redundant code and stale comments
- drop search
- Correct what a `vectorize` that avoided `Search` would need
- Record why `Search` cannot be replaced by a calling convention
- Try fearless_simd#347: the trampoline goes where the level is baseline
- Answer a short haystack a byte at a time
- Give both probes a full `memchr` comparison
- Cover a short haystack with two overlapping reads instead of a walk
- Reach the target-feature context through public API only
- Share the `Scan` behind a `&'static`, and correct what that was meant to buy
- Add an iteration probe; the obvious better ABI for `find_next` is not one
- Move the word-family scan builders into `swar` and `bytewise`
- Move the vector level dispatch into the vector module
- Take the iterator state out of `count_all`; keep the splats, with evidence
- Delete `ByteSet`; the file was always about kinds, not sets
- Drop the generated modules; ask upstream for the attribute directly
- Let `fearless_simd` write the target-feature attributes
- Make `find` beat `memchr` on short haystacks
- Give `find` its own entry point, and stop building the set twice
- Fold `Finder` into `Bytes`, renamed `MemchrN`, fixed at construction
- updates
- pre-store kernel info in a union, and return bit count from indirect call
- drop bounds checks
- 32x16 swizzle
- better swar
- better benches
- vector kernel generic over vector
- misc changes
- move vector kernels from general purpose to vectors
- swar changes
- Stage the vector tail at constant offsets
- Better handle the tail for vector
- cleanup some swar
- cleanup the vector kernels
- docs
- Implement the things
