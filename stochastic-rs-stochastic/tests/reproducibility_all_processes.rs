//! Exhaustive reproducibility guard: every concrete `ProcessExt` implementor
//! in `stochastic-rs-stochastic`, instantiated with a `Deterministic` seed,
//! must produce bit-identical output from two fresh identically-seeded
//! instances, and `sample_par` must be bit-identical across rayon
//! thread-pool sizes at both a size `<= MAX_CHUNKS` (one path per chunk) and
//! a size above it (several paths per chunk).
//!
//! **How this list was derived and how to extend it.** `grep -rn
//! "ProcessExt<T> for\|ProcessExt<T," stochastic-rs-stochastic/src
//! --include='*.rs'` finds 124 concrete implementations outside
//! `src/traits/` (the file `traits/process.rs` contributes only blanket
//! impls of *marker* traits keyed off a `P: ProcessExt<...>` bound —
//! `OneDimensional`, `MultiDimensional`, `TwoDimensional`, `CurveOutput`,
//! `VariableDimensional`, `ComplexPathOutput` — never a concrete process
//! type, so excluding the whole file is correct regardless of exactly how
//! many of those six a naive line-count would credit it with). Per
//! directory: diffusion 34, process 20, jump 17, volatility 15, interest
//! 15, autoregressive 9, noise 5, rough 4, correlation 4, sheet 1 — 124
//! total, split below into one submodule per directory purely to keep every
//! file under this crate's line-count limit (`common` holds the shared
//! `check`/`guard!`/`ReproBits` machinery all ten submodules use; they and
//! this file together compile into one `reproducibility_all_processes` test
//! binary, so `cargo test` runs and reports all 124 checks as before).
//! **When a new process type is added, add one `guard!` line for it** in
//! the submodule matching its source directory — a type with no line
//! anywhere in this tree is a type this guard is not proving anything
//! about.
//!
//! Sixteen types cannot be built from scalars alone (nine are generic over
//! a compile-time backend `B` — instantiated here on the default `Cpu`
//! backend only, since GPU backends deliberately ignore the seed and are
//! not part of this guarantee; seven take a jump-size distribution `D`,
//! passed as `ScalarNormal`/`ScalarExp` per the crate's convention) — these
//! get the same one-closure-per-type treatment as everything else, just
//! with a longer argument list. `process::ccustom::CompoundCustom` takes
//! *two* such distributions plus a nested `CustomJt`, an eighth
//! distribution-shaped case beyond the seven usually named; it is covered
//! in `process.rs` alongside the rest. `n` is kept small throughout (this
//! guard is about seed plumbing, not statistics); jump intensities are
//! pushed high enough that a jump component's own reproducibility bug
//! cannot hide behind a diffusion-only comparison.
//!
//! Bit-identity always compares two **freshly constructed** identically-
//! seeded instances, never two `.sample()` calls on one object — `sampler()`
//! re-derives the seed per call, so repeated sampling on one object
//! legitimately diverges (see `ProcessExt`'s own "`Clone` semantics" doc).

#[path = "reproducibility_all_processes/common.rs"]
mod common;

#[path = "reproducibility_all_processes/autoregressive.rs"]
mod autoregressive;
#[path = "reproducibility_all_processes/correlation.rs"]
mod correlation;
#[path = "reproducibility_all_processes/diffusion.rs"]
mod diffusion;
#[path = "reproducibility_all_processes/interest.rs"]
mod interest;
#[path = "reproducibility_all_processes/jump.rs"]
mod jump;
#[path = "reproducibility_all_processes/noise.rs"]
mod noise;
#[path = "reproducibility_all_processes/process.rs"]
mod process;
#[path = "reproducibility_all_processes/rough.rs"]
mod rough;
#[path = "reproducibility_all_processes/sheet.rs"]
mod sheet;
#[path = "reproducibility_all_processes/volatility.rs"]
mod volatility;
