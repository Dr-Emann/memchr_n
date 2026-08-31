//! Times the public API on haystacks far larger than last-level cache.
//!
//! The scanning loop is wide enough that its cost per byte changes with where the
//! haystack lives; `bench_probe`'s haystack is cache-resident, so this covers the other
//! end.

use memchr_n::{Bytes, Finder};
use std::hint::black_box;
use std::time::Instant;

const SEED: &[u8] = include_bytes!("../benches/haystacks/sherlock/huge.txt");

/// Never occurs in the text, so the scan runs the whole way through.
const NEVER: u8 = b'<';
const RARE: u8 = b'z';

fn finder(needles: &[u8]) -> Finder {
    Bytes::from_bytes(needles).finder()
}

const ROUNDS: u32 = 100;

fn best<T>(rounds: u32, mut f: impl FnMut() -> T) -> f64 {
    // The first pass over a fresh mapping pays for its page faults, and at these sizes
    // that dwarfs the scan itself.
    black_box(f());
    let mut best = f64::MAX;
    for _ in 0..rounds {
        let start = Instant::now();
        black_box(f());
        best = best.min(start.elapsed().as_secs_f64());
    }
    best
}

fn row(label: &str, len: usize, ours: f64, theirs: f64) {
    let gbs = |t: f64| len as f64 / t / 1e9;
    println!(
        "{label:22} {:9.1} us {:6.1} GB/s   memchr {:9.1} us {:6.1} GB/s   {:5.2}x",
        ours * 1e6,
        gbs(ours),
        theirs * 1e6,
        gbs(theirs),
        theirs / ours,
    );
}

fn main() {
    for mb in [1usize, 4, 16, 64, 256] {
        let len = mb * 1024 * 1024;
        let mut haystack = Vec::with_capacity(len);
        while haystack.len() < len {
            let take = (len - haystack.len()).min(SEED.len());
            haystack.extend_from_slice(&SEED[..take]);
        }
        let never = finder(&[NEVER]);
        let rare = finder(&[RARE]);

        let scan = best(ROUNDS, || never.find(black_box(&haystack)));
        let scan_theirs = best(ROUNDS, || memchr::memchr(NEVER, black_box(&haystack)));
        // `count` does not go through the widened loop, so it holds still between builds
        // and shows how much of any difference is the machine rather than the code.
        let count = best(ROUNDS, || rare.iter(black_box(&haystack)).count());
        let count_theirs = best(ROUNDS, || {
            memchr::memchr_iter(RARE, black_box(&haystack)).count()
        });

        row(&format!("{mb:4} MB  find(never)"), len, scan, scan_theirs);
        row(&format!("{mb:4} MB  count(rare)"), len, count, count_theirs);
    }
}
