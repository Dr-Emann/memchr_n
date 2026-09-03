//! Splits a one-shot search into the two things it pays for that a prebuilt one does not.
//!
//! `build` is a construction with nothing done to it. `prebuilt` is one search through a
//! `MemchrN` that already exists, which is the floor. `oneshot` is construction and search
//! together, so `oneshot - prebuilt - build` is what the two cost when neither can be
//! optimised against the other — mostly the indirect call standing between them.
//!
//! Three haystacks per length. `hit` matches at offset 0, so the search returns as early as it
//! can and the row is almost all fixed cost. `mid` matches halfway, which is what a scan that
//! answers a byte at a time has to pay for the offsets it walks past. `miss` never matches, so
//! the row is the whole scan.

use memchr_n::MemchrN;
use std::hint::black_box;
use std::time::Instant;

const ROUNDS: u32 = 200;
const ITERS: u32 = 500;

const LENS: [usize; 12] = [1, 4, 8, 12, 16, 32, 64, 128, 512, 4096, 65536, 1 << 20];

/// The kinds a set can resolve to, one set each.
const SETS: [(&str, &[u8]); 7] = [
    ("one-byte", b"z"),
    ("two-bytes", b"yz"),
    ("three-bytes", b"xyz"),
    ("one-range", b"0123456789"),
    ("small-set", b"aeiouAEI"),
    ("single-nibble", b"abcdefghjl"),
    ("any-byte", b"0123456789abcdef"),
];

/// The same search through `memchr`, for the sets it can spell.
///
/// It tops out at three needles and has no prebuilt form in its portable API, so this is
/// their whole call against our prebuilt one — the comparison in their favour.
fn theirs(needles: &[u8], hay: &[u8]) -> f64 {
    match *needles {
        [a] => best(|| memchr::memchr(black_box(a), black_box(hay))),
        [a, b] => best(|| memchr::memchr2(black_box(a), black_box(b), black_box(hay))),
        [a, b, c] => best(|| memchr::memchr3(black_box(a), black_box(b), black_box(c), black_box(hay))),
        _ => f64::NAN,
    }
}

fn best<T>(mut f: impl FnMut() -> T) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..ROUNDS {
        let start = Instant::now();
        for _ in 0..ITERS {
            black_box(f());
        }
        best = best.min(start.elapsed().as_secs_f64() / f64::from(ITERS));
    }
    best
}

/// Where the one match in a haystack sits, if there is one.
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

/// A haystack of `len` bytes that no set above matches, with `needles[0]` planted where
/// `planted` says.
fn haystack(len: usize, needles: &[u8], planted: Planted) -> Vec<u8> {
    let mut hay = vec![b'.'; len];
    let offset = match planted {
        Planted::Front => 0,
        Planted::Middle => len / 2,
        Planted::Nowhere => return hay,
    };
    if offset < len {
        hay[offset] = needles[0];
    }
    hay
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
        let build = best(|| MemchrN::new(black_box(needles)));
        let prebuilt_finder = MemchrN::new(needles);

        for planted in [Planted::Front, Planted::Middle, Planted::Nowhere] {
            for len in LENS {
                let hay = haystack(len, needles, planted);
                let hay = hay.as_slice();

                let prebuilt = best(|| black_box(&prebuilt_finder).iter(black_box(hay)).next());
                let direct = best(|| black_box(&prebuilt_finder).find(black_box(hay)));
                let oneshot = best(|| MemchrN::new(black_box(needles)).find(black_box(hay)));
                let theirs = theirs(needles, hay);

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
