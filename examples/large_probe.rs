//! Times the public API on haystacks far larger than last-level cache.
//!
//! The scanning loop is wide enough that its cost per byte changes with where the
//! haystack lives; `bench_probe`'s haystack is cache-resident, so this covers the other
//! end.

use memchr_n::{Bytes, Finder};
use std::hint::black_box;
use std::time::Instant;

const SEED: &[u8] = include_bytes!("../benches/haystacks/sherlock/huge.txt");

fn finder(needles: &[u8]) -> Finder {
    Bytes::from_bytes(needles).finder()
}

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

fn main() {
    for mb in [1usize, 4, 16, 64, 256] {
        let len = mb * 1024 * 1024;
        let mut haystack = Vec::with_capacity(len);
        while haystack.len() < len {
            let take = (len - haystack.len()).min(SEED.len());
            haystack.extend_from_slice(&SEED[..take]);
        }
        // `<` never occurs in the text, so the scan runs the whole way through.
        let never = finder(b"<");
        let rare = finder(b"z");

        // `count` does not go through the widened loop, so it holds still between builds
        // and shows how much of any difference is the machine rather than the code.
        let scan = best(20, || never.find(black_box(&haystack)));
        let count = best(20, || rare.iter(black_box(&haystack)).count());
        let gbs = |t: f64| len as f64 / t / 1e9;
        println!(
            "{mb:4} MB   find(never) {:9.1} us {:6.1} GB/s    count(rare) {:9.1} us {:6.1} GB/s",
            scan * 1e6,
            gbs(scan),
            count * 1e6,
            gbs(count),
        );
    }
}
