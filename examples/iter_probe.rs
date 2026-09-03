//! Iteration throughput, as the minimum of many rounds rather than a mean.
//!
//! `iterate` in the benchmark suite measures the same thing, but a mean over a noisy host
//! moves further between runs of one binary than the changes worth seeing move it.

use memchr_n::MemchrN;
use std::hint::black_box;
use std::time::Instant;

const SHERLOCK: &[u8] = include_bytes!("../benches/haystacks/sherlock/huge.txt");

const ROUNDS: u32 = 60;

/// By match density, sparse first. All are one to three needles, so `memchr` can spell every
/// one of them.
const SETS: [(&str, &[u8]); 6] = [
    ("never1", b"<"),
    ("rare1", b"z"),
    ("rare3", b"zRJ"),
    ("uncommon1", b"b"),
    ("common3", b"ato"),
    ("verycommon1", b" "),
];

/// `memchr`'s own search for the same set, which is a whole call — that crate has no
/// prebuilt form in its portable API.
fn their_find(needles: &[u8], hay: &[u8]) -> Option<usize> {
    match *needles {
        [a] => memchr::memchr(a, hay),
        [a, b] => memchr::memchr2(a, b, hay),
        [a, b, c] => memchr::memchr3(a, b, c, hay),
        _ => unreachable!("every set above is one to three needles"),
    }
}

/// `memchr`'s iterator over the same set.
///
/// Boxed, which costs an allocation per call — negligible against the microseconds a whole
/// pass over the corpus takes, but not against a single `find`, which is why that one goes
/// through [`their_find`] instead.
fn their_iter<'h>(needles: &[u8], hay: &'h [u8]) -> Box<dyn Iterator<Item = usize> + 'h> {
    match *needles {
        [a] => Box::new(memchr::memchr_iter(a, hay)),
        [a, b] => Box::new(memchr::memchr2_iter(a, b, hay)),
        [a, b, c] => Box::new(memchr::memchr3_iter(a, b, c, hay)),
        _ => unreachable!("every set above is one to three needles"),
    }
}

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
    println!(
        "level {:?}, sherlock {} bytes, us / memchr",
        fearless_simd::Level::new(),
        SHERLOCK.len()
    );
    println!(
        "{:>13} {:>9} {:>9}   {:>9} {:>9}   {:>9} {:>9}",
        "set", "iterate", "theirs", "count", "theirs", "find", "theirs"
    );
    for (name, needles) in SETS {
        let finder = MemchrN::new(needles);
        let hay = SHERLOCK;

        let iterate = best(|| {
            black_box(&finder)
                .iter(black_box(hay))
                .fold(0usize, |acc, off| acc.wrapping_add(off))
        });
        let their_iterate = best(|| {
            their_iter(needles, black_box(hay)).fold(0usize, |acc, off| acc.wrapping_add(off))
        });
        let count = best(|| black_box(&finder).iter(black_box(hay)).count());
        let their_count = best(|| their_iter(needles, black_box(hay)).count());
        let find = best(|| black_box(&finder).find(black_box(hay)).unwrap_or(0));
        let their_find = best(|| their_find(needles, black_box(hay)).unwrap_or(0));

        println!(
            "{name:>13} {iterate:>9.2} {their_iterate:>9.2}   \
             {count:>9.2} {their_count:>9.2}   {find:>9.4} {their_find:>9.4}"
        );
    }
}
