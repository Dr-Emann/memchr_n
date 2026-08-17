use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use memchr_n::{Bytes, Finder};
use std::hint::black_box;

const SHERLOCK_TINY: &[u8] = include_bytes!("haystacks/sherlock/tiny.txt");
const SHERLOCK_SMALL: &[u8] = include_bytes!("haystacks/sherlock/small.txt");
const SHERLOCK_HUGE: &[u8] = include_bytes!("haystacks/sherlock/huge.txt");
const SUBTITLES_EN: &[u8] = include_bytes!("haystacks/opensubtitles/en-huge.txt");
const SUBTITLES_RU: &[u8] = include_bytes!("haystacks/opensubtitles/ru-huge.txt");
const SUBTITLES_ZH: &[u8] = include_bytes!("haystacks/opensubtitles/zh-huge.txt");
const CODE_RUST: &[u8] = include_bytes!("haystacks/code/rust-library.rs");
const MD5: &[u8] = include_bytes!("haystacks/pathological/md5-huge.txt");
const RANDOM: &[u8] = include_bytes!("haystacks/pathological/random-huge.txt");

const ALNUM: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const HEX_LOWER: &[u8] = b"0123456789abcdef";

const OURS: &str = "memchr_n";
const THEIRS: &str = "memchr";

#[derive(Copy, Clone)]
enum ByteSet {
    List(&'static [u8]),
    Range(u8, u8),
}

impl ByteSet {
    fn finder(self) -> Finder {
        let bitset = match self {
            ByteSet::List(bytes) => Bytes::from_bytes(bytes),
            ByteSet::Range(start, end) => Bytes::from_range(start..=end),
        };
        bitset.finder()
    }

    fn contains(self, byte: u8) -> bool {
        match self {
            ByteSet::List(bytes) => bytes.contains(&byte),
            ByteSet::Range(start, end) => start <= byte && byte <= end,
        }
    }
}

/// The subset of byte sets the `memchr` crate can also express, via `memchr`/`memchr2`/`memchr3`.
#[derive(Copy, Clone)]
enum Needles {
    One(u8),
    Two(u8, u8),
    Three(u8, u8, u8),
}

impl Needles {
    fn from_set(set: ByteSet) -> Option<Needles> {
        let ByteSet::List(bytes) = set else {
            return None;
        };
        match *bytes {
            [n1] => Some(Needles::One(n1)),
            [n1, n2] => Some(Needles::Two(n1, n2)),
            [n1, n2, n3] => Some(Needles::Three(n1, n2, n3)),
            _ => None,
        }
    }

    fn count(self, haystack: &[u8]) -> usize {
        match self {
            Needles::One(n1) => memchr::memchr_iter(n1, haystack).count(),
            Needles::Two(n1, n2) => memchr::memchr2_iter(n1, n2, haystack).count(),
            Needles::Three(n1, n2, n3) => memchr::memchr3_iter(n1, n2, n3, haystack).count(),
        }
    }

    fn first(self, haystack: &[u8]) -> Option<usize> {
        match self {
            Needles::One(n1) => memchr::memchr(n1, haystack),
            Needles::Two(n1, n2) => memchr::memchr2(n1, n2, haystack),
            Needles::Three(n1, n2, n3) => memchr::memchr3(n1, n2, n3, haystack),
        }
    }

    fn offset_sum(self, haystack: &[u8]) -> usize {
        let mut sum = 0usize;
        match self {
            Needles::One(n1) => {
                for offset in memchr::memchr_iter(n1, haystack) {
                    sum = sum.wrapping_add(offset);
                }
            }
            Needles::Two(n1, n2) => {
                for offset in memchr::memchr2_iter(n1, n2, haystack) {
                    sum = sum.wrapping_add(offset);
                }
            }
            Needles::Three(n1, n2, n3) => {
                for offset in memchr::memchr3_iter(n1, n2, n3, haystack) {
                    sum = sum.wrapping_add(offset);
                }
            }
        }
        sum
    }
}

/// Byte sets ordered by match density, mirroring the `memchr` crate's sherlock benchmarks.
/// All of them are 1-3 bytes so that every entry has a `memchr` counterpart.
const DENSITY_SETS: &[(&str, ByteSet)] = &[
    ("never1", ByteSet::List(b"<")),
    ("rare1", ByteSet::List(b"z")),
    ("rare2", ByteSet::List(b"zR")),
    ("rare3", ByteSet::List(b"zRJ")),
    ("uncommon1", ByteSet::List(b"b")),
    ("uncommon3", ByteSet::List(b"bp.")),
    ("common1", ByteSet::List(b"a")),
    ("common3", ByteSet::List(b"ato")),
    ("verycommon1", ByteSet::List(b" ")),
];

/// One byte set per [`Finder`] specialization, so each SIMD kernel is measured directly.
const KIND_SETS: &[(&str, ByteSet)] = &[
    ("never", ByteSet::List(b"")),
    ("one-byte", ByteSet::List(b"z")),
    ("one-range", ByteSet::Range(b'0', b'9')),
    ("small-set", ByteSet::List(b"aeiouAEI")),
    ("single-nibble", ByteSet::List(b"abcdefghjl")),
    ("any-byte-16", ByteSet::List(HEX_LOWER)),
    ("any-byte-62", ByteSet::List(ALNUM)),
];

const CORPORA: &[(&str, &[u8])] = &[
    ("sherlock", SHERLOCK_HUGE),
    ("subtitles-en", SUBTITLES_EN),
    ("subtitles-ru", SUBTITLES_RU),
    ("subtitles-zh", SUBTITLES_ZH),
    ("code-rust", CODE_RUST),
    ("md5", MD5),
    ("random", RANDOM),
];

const CORPUS_SETS: &[(&str, ByteSet)] = &[
    ("space", ByteSet::List(b" ")),
    ("nonascii", ByteSet::Range(0x80, 0xFF)),
    ("alnum", ByteSet::List(ALNUM)),
];

const SIZE_SETS: &[(&str, ByteSet)] = &[
    ("rare1", ByteSet::List(b"z")),
    ("common1", ByteSet::List(b"a")),
];

const SIZES: &[(&str, &[u8])] = &[
    ("empty", b""),
    ("tiny", SHERLOCK_TINY),
    ("small", SHERLOCK_SMALL),
    ("huge", SHERLOCK_HUGE),
];

fn naive_count(set: ByteSet, haystack: &[u8]) -> usize {
    let mut count = 0;
    for &byte in haystack {
        if set.contains(byte) {
            count += 1;
        }
    }
    count
}

fn naive_offset_sum(set: ByteSet, haystack: &[u8]) -> usize {
    let mut sum = 0usize;
    for (offset, &byte) in haystack.iter().enumerate() {
        if set.contains(byte) {
            sum = sum.wrapping_add(offset);
        }
    }
    sum
}

fn naive_first(set: ByteSet, haystack: &[u8]) -> Option<usize> {
    for (offset, &byte) in haystack.iter().enumerate() {
        if set.contains(byte) {
            return Some(offset);
        }
    }
    None
}

fn offset_sum(finder: &Finder, haystack: &[u8]) -> usize {
    let mut sum = 0usize;
    for offset in finder.iter(haystack) {
        sum = sum.wrapping_add(offset);
    }
    sum
}

/// Guards against benchmarking a broken implementation, which would otherwise show up
/// as a suspiciously fast result rather than a failure.
fn verified_finder(set: ByteSet, haystack: &[u8], label: &str) -> Finder {
    let finder = set.finder();
    assert_eq!(
        finder.iter(haystack).count(),
        naive_count(set, haystack),
        "count mismatch for {label}"
    );
    assert_eq!(
        finder.iter(haystack).next(),
        naive_first(set, haystack),
        "first-match mismatch for {label}"
    );
    assert_eq!(
        offset_sum(&finder, haystack),
        naive_offset_sum(set, haystack),
        "offset mismatch for {label}"
    );
    finder
}

/// Returns [`None`] for sets `memchr` cannot express, so those benchmarks run
/// unpaired rather than being dropped.
fn verified_needles(set: ByteSet, haystack: &[u8], label: &str) -> Option<Needles> {
    let needles = Needles::from_set(set)?;
    assert_eq!(
        needles.count(haystack),
        naive_count(set, haystack),
        "memchr count mismatch for {label}"
    );
    assert_eq!(
        needles.first(haystack),
        naive_first(set, haystack),
        "memchr first-match mismatch for {label}"
    );
    assert_eq!(
        needles.offset_sum(haystack),
        naive_offset_sum(set, haystack),
        "memchr offset mismatch for {label}"
    );
    Some(needles)
}

/// Both crates get their fastest public counting path here, which is not the same
/// algorithm on each side: `memchr_iter().count()` is specialized, but
/// `memchr2_iter`/`memchr3_iter` fall back to walking match by match, so their numbers
/// degrade with match density. The `iterate` group compares the per-match paths.
fn bench_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("count/sherlock");
    group.throughput(Throughput::Bytes(SHERLOCK_HUGE.len() as u64));
    for &(name, set) in DENSITY_SETS {
        let finder = verified_finder(set, SHERLOCK_HUGE, name);
        group.bench_with_input(BenchmarkId::new(OURS, name), &finder, |b, finder| {
            b.iter(|| black_box(finder).iter(black_box(SHERLOCK_HUGE)).count())
        });
        let Some(needles) = verified_needles(set, SHERLOCK_HUGE, name) else {
            continue;
        };
        group.bench_with_input(BenchmarkId::new(THEIRS, name), &needles, |b, &needles| {
            b.iter(|| black_box(needles).count(black_box(SHERLOCK_HUGE)))
        });
    }
    group.finish();
}

fn bench_find_first(c: &mut Criterion) {
    let mut group = c.benchmark_group("find-first/sherlock");
    group.throughput(Throughput::Bytes(SHERLOCK_HUGE.len() as u64));
    for &(name, set) in DENSITY_SETS {
        let finder = verified_finder(set, SHERLOCK_HUGE, name);
        group.bench_with_input(BenchmarkId::new(OURS, name), &finder, |b, finder| {
            b.iter(|| black_box(finder).iter(black_box(SHERLOCK_HUGE)).next())
        });
        let Some(needles) = verified_needles(set, SHERLOCK_HUGE, name) else {
            continue;
        };
        group.bench_with_input(BenchmarkId::new(THEIRS, name), &needles, |b, &needles| {
            b.iter(|| black_box(needles).first(black_box(SHERLOCK_HUGE)))
        });
    }
    group.finish();
}

fn bench_iterate(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterate/sherlock");
    group.throughput(Throughput::Bytes(SHERLOCK_HUGE.len() as u64));
    for &(name, set) in DENSITY_SETS {
        let finder = verified_finder(set, SHERLOCK_HUGE, name);
        group.bench_with_input(BenchmarkId::new(OURS, name), &finder, |b, finder| {
            b.iter(|| offset_sum(black_box(finder), black_box(SHERLOCK_HUGE)))
        });
        let Some(needles) = verified_needles(set, SHERLOCK_HUGE, name) else {
            continue;
        };
        group.bench_with_input(BenchmarkId::new(THEIRS, name), &needles, |b, &needles| {
            b.iter(|| black_box(needles).offset_sum(black_box(SHERLOCK_HUGE)))
        });
    }
    group.finish();
}

/// Byte sets with no `memchr` counterpart, so this group measures each [`Finder`]
/// specialization on its own.
fn bench_kinds(c: &mut Criterion) {
    let mut group = c.benchmark_group("kind/sherlock");
    group.throughput(Throughput::Bytes(SHERLOCK_HUGE.len() as u64));
    for &(name, set) in KIND_SETS {
        let finder = verified_finder(set, SHERLOCK_HUGE, name);
        group.bench_function(name, |b| {
            b.iter(|| black_box(&finder).iter(black_box(SHERLOCK_HUGE)).count())
        });
    }
    group.finish();
}

fn bench_corpora(c: &mut Criterion) {
    let mut group = c.benchmark_group("count/corpora");
    for &(corpus, haystack) in CORPORA {
        group.throughput(Throughput::Bytes(haystack.len() as u64));
        for &(name, set) in CORPUS_SETS {
            let param = format!("{name}/{corpus}");
            let finder = verified_finder(set, haystack, &param);
            group.bench_with_input(BenchmarkId::new(OURS, &param), &finder, |b, finder| {
                b.iter(|| black_box(finder).iter(black_box(haystack)).count())
            });
            let Some(needles) = verified_needles(set, haystack, &param) else {
                continue;
            };
            group.bench_with_input(BenchmarkId::new(THEIRS, &param), &needles, |b, &needles| {
                b.iter(|| black_box(needles).count(black_box(haystack)))
            });
        }
    }
    group.finish();
}

/// Per-call overhead as the haystack shrinks; deliberately reports latency rather than
/// throughput, since that is what dominates for short inputs.
fn bench_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("count/sizes");
    for &(name, set) in SIZE_SETS {
        for &(size, haystack) in SIZES {
            let param = format!("{name}/{size}");
            let finder = verified_finder(set, haystack, &param);
            group.bench_with_input(BenchmarkId::new(OURS, &param), &finder, |b, finder| {
                b.iter(|| black_box(finder).iter(black_box(haystack)).count())
            });
            let Some(needles) = verified_needles(set, haystack, &param) else {
                continue;
            };
            group.bench_with_input(BenchmarkId::new(THEIRS, &param), &needles, |b, &needles| {
                b.iter(|| black_box(needles).count(black_box(haystack)))
            });
        }
    }
    group.finish();
}

/// `memchr` has no counterpart here: its equivalent prebuilt finders live behind
/// arch-gated modules rather than the portable public API.
fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("build");
    for &(name, set) in KIND_SETS {
        group.bench_function(name, |b| b.iter(|| black_box(set).finder()));
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_count,
    bench_find_first,
    bench_iterate,
    bench_kinds,
    bench_corpora,
    bench_sizes,
    bench_build,
);
criterion_main!(benches);
