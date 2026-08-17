//! Measures scans on haystacks larger than the last-level cache.

mod timing;

use memchr_n::MemchrN;
use std::hint::black_box;
use timing::best;

const SEED: &[u8] = include_bytes!("../benches/haystacks/sherlock/huge.txt");

const NEVER: u8 = b'<';
const RARE: u8 = b'z';

fn finder(needles: &[u8]) -> MemchrN {
    MemchrN::new(needles)
}

const ROUNDS: u32 = 100;

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

fn timed<T>(rounds: u32, mut f: impl FnMut() -> T) -> f64 {
    // Exclude the first pass, which includes page faults.
    black_box(f());
    best(rounds, 1, f)
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

        let scan = timed(ROUNDS, || never.find(black_box(&haystack)));
        let scan_theirs = timed(ROUNDS, || memchr::memchr(NEVER, black_box(&haystack)));
        let count = timed(ROUNDS, || rare.iter(black_box(&haystack)).count());
        let count_theirs = timed(ROUNDS, || {
            memchr::memchr_iter(RARE, black_box(&haystack)).count()
        });

        row(&format!("{mb:4} MB  find(never)"), len, scan, scan_theirs);
        row(&format!("{mb:4} MB  count(rare)"), len, count, count_theirs);
    }
}
