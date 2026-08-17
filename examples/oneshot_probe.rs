//! Separates construction, prebuilt search, and combined one-shot search costs.
//!
//! `overhead` is `oneshot - build - find`.

mod timing;

use memchr_n::MemchrN;
use std::hint::black_box;
use timing::best;

const ROUNDS: u32 = 200;
const ITERS: u32 = 500;

const LENS: [usize; 12] = [1, 4, 8, 12, 16, 32, 64, 128, 512, 4096, 65536, 1 << 20];

const SETS: [(&str, &[u8]); 7] = [
    ("one-byte", b"z"),
    ("two-bytes", b"yz"),
    ("three-bytes", b"xyz"),
    ("one-range", b"0123456789"),
    ("small-set", b"aeiouAEI"),
    ("single-nibble", b"abcdefghjl"),
    ("any-byte", b"0123456789abcdef"),
];

fn theirs(needles: &[u8], haystack: &[u8]) -> f64 {
    match *needles {
        [a] => best(ROUNDS, ITERS, || {
            memchr::memchr(black_box(a), black_box(haystack))
        }),
        [a, b] => best(ROUNDS, ITERS, || {
            memchr::memchr2(black_box(a), black_box(b), black_box(haystack))
        }),
        [a, b, c] => best(ROUNDS, ITERS, || {
            memchr::memchr3(
                black_box(a),
                black_box(b),
                black_box(c),
                black_box(haystack),
            )
        }),
        _ => f64::NAN,
    }
}

#[derive(Copy, Clone)]
enum Planted {
    Front,
    Middle,
    Nowhere,
}

impl Planted {
    fn label(self) -> &'static str {
        match self {
            Planted::Front => "hit",
            Planted::Middle => "mid",
            Planted::Nowhere => "miss",
        }
    }
}

fn haystack(len: usize, needles: &[u8], planted: Planted) -> Vec<u8> {
    let mut haystack = vec![b'.'; len];
    let offset = match planted {
        Planted::Front => 0,
        Planted::Middle => len / 2,
        Planted::Nowhere => return haystack,
    };
    if offset < len {
        haystack[offset] = needles[0];
    }
    haystack
}

fn main() {
    println!(
        "level {:?}, MemchrN {} bytes",
        fearless_simd::Level::new(),
        size_of::<MemchrN>()
    );
    println!(
        "{:>14} {:>12} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "set", "len", "build", "via-iter", "find", "oneshot", "overhead", "memchr"
    );

    for (name, needles) in SETS {
        let build = best(ROUNDS, ITERS, || MemchrN::new(black_box(needles)));
        let prebuilt_finder = MemchrN::new(needles);

        for planted in [Planted::Front, Planted::Middle, Planted::Nowhere] {
            for len in LENS {
                let haystack = haystack(len, needles, planted);
                let haystack = haystack.as_slice();

                let prebuilt = best(ROUNDS, ITERS, || {
                    black_box(&prebuilt_finder).iter(black_box(haystack)).next()
                });
                let direct = best(ROUNDS, ITERS, || {
                    black_box(&prebuilt_finder).find(black_box(haystack))
                });
                let oneshot = best(ROUNDS, ITERS, || {
                    MemchrN::new(black_box(needles)).find(black_box(haystack))
                });
                let theirs = theirs(needles, haystack);

                let label = planted.label();
                println!(
                    "{:>14} {:>7}/{:<4} {:>7.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1}",
                    name,
                    len,
                    label,
                    build * 1e9,
                    prebuilt * 1e9,
                    direct * 1e9,
                    oneshot * 1e9,
                    (oneshot - direct - build) * 1e9,
                    theirs * 1e9,
                );
            }
        }
    }
}
