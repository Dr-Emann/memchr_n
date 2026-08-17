//! Shared timing support for probe examples.

use std::hint::black_box;
use std::time::Instant;

/// Returns the fastest observed seconds per call to reduce scheduler noise.
pub fn best<T>(rounds: u32, iters: u32, mut f: impl FnMut() -> T) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        best = best.min(start.elapsed().as_secs_f64() / f64::from(iters));
    }
    best
}
