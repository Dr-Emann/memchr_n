//! Measures iteration throughput using the fastest timed round.

mod timing;

use memchr_n::MemchrN;
use std::hint::black_box;
use timing::best;

const SHERLOCK: &[u8] = include_bytes!("../benches/haystacks/sherlock/huge.txt");

const ROUNDS: u32 = 60;

const SETS: [(&str, &[u8]); 6] = [
    ("never1", b"<"),
    ("rare1", b"z"),
    ("rare3", b"zRJ"),
    ("uncommon1", b"b"),
    ("common3", b"ato"),
    ("verycommon1", b" "),
];

fn their_find(needles: &[u8], haystack: &[u8]) -> Option<usize> {
    match *needles {
        [a] => memchr::memchr(a, haystack),
        [a, b] => memchr::memchr2(a, b, haystack),
        [a, b, c] => memchr::memchr3(a, b, c, haystack),
        _ => unreachable!("every set above is one to three needles"),
    }
}

// Boxing is negligible over a full corpus pass.
fn their_iter<'h>(needles: &[u8], haystack: &'h [u8]) -> Box<dyn Iterator<Item = usize> + 'h> {
    match *needles {
        [a] => Box::new(memchr::memchr_iter(a, haystack)),
        [a, b] => Box::new(memchr::memchr2_iter(a, b, haystack)),
        [a, b, c] => Box::new(memchr::memchr3_iter(a, b, c, haystack)),
        _ => unreachable!("every set above is one to three needles"),
    }
}

fn micros(f: impl FnMut() -> usize) -> f64 {
    best(ROUNDS, 1, f) * 1e6
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
        let haystack = SHERLOCK;

        let iterate = micros(|| {
            black_box(&finder)
                .iter(black_box(haystack))
                .fold(0usize, |acc, off| acc.wrapping_add(off))
        });
        let their_iterate = micros(|| {
            their_iter(needles, black_box(haystack)).fold(0usize, |acc, off| acc.wrapping_add(off))
        });
        let count = micros(|| black_box(&finder).iter(black_box(haystack)).count());
        let their_count = micros(|| their_iter(needles, black_box(haystack)).count());
        let find = micros(|| black_box(&finder).find(black_box(haystack)).unwrap_or(0));
        let their_find = micros(|| their_find(needles, black_box(haystack)).unwrap_or(0));

        println!(
            "{name:>13} {iterate:>9.2} {their_iterate:>9.2}   \
             {count:>9.2} {their_count:>9.2}   {find:>9.4} {their_find:>9.4}"
        );
    }
}
