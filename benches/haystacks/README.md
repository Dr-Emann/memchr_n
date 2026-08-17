# Benchmark corpora

These files were copied from the `memchr` crate's benchmark suite
(<https://github.com/BurntSushi/memchr>, `benchmarks/haystacks`), so results here
are comparable with the numbers that crate publishes. Each sub-directory keeps its
original README.

Only the subset used by `benches/memchr_n.rs` was copied:

- `sherlock/` — the classic memchr corpus, at three sizes.
- `opensubtitles/` — English, Russian and Chinese subtitle text, for measuring
  byte-set searches over non-ASCII input.
- `code/rust-library.rs` — concatenated Rust source, a stand-in for code search.
- `pathological/md5-huge.txt` — one md5 hash per line, so no byte is much more
  frequent than any other.
- `pathological/random-huge.txt` — hex-encoded output of `/dev/urandom`.
