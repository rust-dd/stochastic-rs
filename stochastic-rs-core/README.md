[![Crates.io](https://img.shields.io/crates/v/stochastic-rs-core?style=flat-square)](https://crates.io/crates/stochastic-rs-core)
[![docs.rs](https://img.shields.io/docsrs/stochastic-rs-core?style=flat-square)](https://docs.rs/stochastic-rs-core)
![License](https://img.shields.io/crates/l/stochastic-rs-core?style=flat-square)

# stochastic-rs-core

**Core traits, SIMD RNG, and seeding for stochastic-rs**

The foundation crate. Everything else in the workspace draws its randomness
from here.

## What is in it

- **`SimdRng`** — a xoshiro-based generator with SIMD bulk fills
  (`fill_uniform_f64`, `fill_ziggurat`, …). Implements `rand::RngCore`, so it
  drops into anything expecting an `Rng`.
- **`SeedExt`** — the workspace seeding contract. Two implementations:
  `Unseeded` (a fresh, globally unique stream per construction, zero
  overhead) and `Deterministic::new(seed)` (reproducible, `AtomicU64`-backed
  so `derive()` can fan out independent child streams from one source).
- **`SimdRngDual`** — an experimental two-engine variant behind the
  `dual-stream-rng` feature. Roughly 5–11% faster on ziggurat-based bulk
  fills on Apple Silicon; its stream is *not* bit-compatible with `SimdRng`.

## Usage

```rust
use stochastic_rs_core::simd_rng::{Deterministic, SimdRng, Unseeded};

// Reproducible
let mut rng = SimdRng::from_seed(42);
let x = rng.next_f64();

// A seed source to hand to a distribution or process
let seed = Deterministic::new(42);
```

## Seeding rule

The seed belongs to the **constructor**, not to an `Rng` you pass in later.
Distributions in this workspace own their stream; a `fill_slice(rng, out)`
call ignores the `rng` argument by design.

## Part of stochastic-rs

This crate is one of the sub-crates of
[**stochastic-rs**](https://github.com/rust-dd/stochastic-rs). Most users
should depend on the umbrella crate, which re-exports everything:

```toml
[dependencies]
stochastic-rs = "3.0.0-beta.2"
```

Depend on `stochastic-rs-core` directly only when you want this slice and nothing else.

- Documentation: [stochastic.rust-dd.com](https://stochastic.rust-dd.com)
- API reference: [docs.rs/stochastic-rs-core](https://docs.rs/stochastic-rs-core)

## License

MIT
