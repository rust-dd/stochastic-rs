[![Crates.io](https://img.shields.io/crates/v/stochastic-rs-ai?style=flat-square)](https://crates.io/crates/stochastic-rs-ai)
[![docs.rs](https://img.shields.io/docsrs/stochastic-rs-ai?style=flat-square)](https://docs.rs/stochastic-rs-ai)
![License](https://img.shields.io/crates/l/stochastic-rs-ai?style=flat-square)

# stochastic-rs-ai

**Neural-network volatility surrogates**

> **Experimental.** This crate is outside the stability promise of stochastic-rs 3.x: its API, the saved-model format and the candle types in its signatures (`Device`) can change in any release. It ships no pretrained weights.

Trained networks that replace an expensive pricing routine with a
sub-millisecond forward pass.

## What is in it

- **Surrogates** — Heston, one-factor Bergomi and rough Bergomi implied
  volatility surfaces on one fixed grid of 11 strikes and 8 maturities
  (`volatility::grid`).
- **`StochVolModelSpec`** — the input/output contract shared by every
  surrogate.
- **Training** — gzip-npy training set loading, a candle-backed network with
  its input and output scaling built in, and a train / save / load round trip.
- **Calibration** — with the `quant` feature, Levenberg–Marquardt on the
  network's exact Jacobian, behind quant's `Calibrator` trait.

With the `quant` feature, `predict_implied_vol_surface` returns quant's `ImpliedVolSurface`:
pass the model's `STRIKES` times the spot, `volatility::grid::MATURITIES` and one forward per
maturity, and the strikes come back ascending.

## Training data

The three gzip-npy training sets the tests use (about 52 MB, in `tests/data`) are
copies of [amuguruza/NN-StochVol-Calibrations](https://github.com/amuguruza/NN-StochVol-Calibrations/tree/master/Data)
(MIT, © 2019 Aitor Muguruza; licence text in `tests/data/LICENSE`). They live in the
repository, not in the crates.io package. The Heston set is indexed by inverse moneyness; see
`volatility::heston::STRIKES`.

## Requirements

Rust 1.89 or newer. On Apple Silicon (or any aarch64 target with the `fp16` target
feature) Rust 1.94 or newer: candle 0.11.0 uses the NEON fp16 intrinsics that stabilised
in 1.94. A candle release that carries
[huggingface/candle#3845](https://github.com/huggingface/candle/pull/3845) lifts this.

## Usage

Enable through the umbrella crate:

```toml
[dependencies]
stochastic-rs = { version = "3.0.0-rc.4", features = ["ai"] }
```

## Part of stochastic-rs

This crate is one of the sub-crates of
[**stochastic-rs**](https://github.com/rust-dd/stochastic-rs). Most users
should depend on the umbrella crate, which re-exports everything:

```toml
[dependencies]
stochastic-rs = "3.0.0-rc.4"
```

Depend on `stochastic-rs-ai` directly only when you want this slice and nothing else.

- Documentation: [stochastic.rust-dd.com](https://stochastic.rust-dd.com)
- Tutorials: [stochastic.rust-dd.com/docs/tutorials](https://stochastic.rust-dd.com/docs/tutorials)
- API reference: [docs.rs/stochastic-rs-ai](https://docs.rs/stochastic-rs-ai)

## License

MIT
