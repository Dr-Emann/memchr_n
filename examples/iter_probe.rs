//! Iteration throughput, as the minimum of many rounds rather than a mean.
//!
//! `iterate` in the benchmark suite measures the same thing, but a mean over a noisy host
//! moves further between runs of one binary than the changes worth seeing move it.

use memchr_n::MemchrN;
use std::hint::black_box;
use std::time::Instant;

const SHERLOCK: &[u8] = include_bytes!("../benches/haystacks/sherlock/huge.txt");

const ROUNDS: u32 = 60;

/// By match density, sparse first.
const SETS: [(&str, &[u8]); 6] = [
    ("never1", b"<"),
    ("rare1", b"z"),
    ("rare3", b"zRJ"),
    ("uncommon1", b"b"),
    ("common3", b"ato"),
    ("verycommon1", b" "),
];

fn best(mut f: impl FnMut() -> usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..ROUNDS {
        let start = Instant::now();
        black_box(f());
        best = best.min(start.elapsed().as_secs_f64());
    }
    best * 1e6
}

fn main() {
    println!("{:>13} {:>10} {:>10} {:>10}", "set", "iterate", "count", "find");
    for (name, needles) in SETS {
        let finder = MemchrN::new(needles);
        let hay = SHERLOCK;

        let iterate = best(|| {
            black_box(&finder)
                .iter(black_box(hay))
                .fold(0usize, |acc, off| acc.wrapping_add(off))
        });
        let count = best(|| black_box(&finder).iter(black_box(hay)).count());
        let find = best(|| black_box(&finder).find(black_box(hay)).unwrap_or(0));

        println!("{name:>13} {iterate:>10.2} {count:>10.2} {find:>10.4}");
    }
}
