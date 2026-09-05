//! The timing loop every probe example runs, in one place.
//!
//! Not an example itself: cargo builds `examples/*.rs` and `examples/*/main.rs`, and a
//! directory holding only a `mod.rs` is neither. That is why this is not `examples/timing.rs`,
//! which cargo would try to build as an example with no `main`.

use std::hint::black_box;
use std::time::Instant;

/// Seconds per call, as the fastest of `rounds` batches of `iters` calls.
///
/// The minimum rather than a mean: on a noisy host a mean moves further between runs than the
/// changes worth seeing move it, and the shortest batch is the one that met the least
/// interference.
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
