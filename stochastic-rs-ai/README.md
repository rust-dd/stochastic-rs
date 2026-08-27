[![Crates.io](https://img.shields.io/crates/v/stochastic-rs-ai?style=flat-square)](https://crates.io/crates/stochastic-rs-ai)
[![docs.rs](https://img.shields.io/docsrs/stochastic-rs-ai?style=flat-square)](https://docs.rs/stochastic-rs-ai)
![License](https://img.shields.io/crates/l/stochastic-rs-ai?style=flat-square)

# stochastic-rs-ai

**Neural-network volatility surrogates**

Trained networks that replace an expensive pricing routine with a
sub-millisecond forward pass.

## What is in it

- **Surrogates** — Heston, one-factor Bergomi and rough Bergomi implied
  volatility surfaces.
- **`StochVolModelSpec`** — the input/output contract shared by every
  surrogate.
- **Scalers** — `BoundedScaler` and `StandardScaler` for pre- and
  post-normalisation.
- **Training** — gzip-npy training set loading, a candle-backed network,
  and a train / save / load round trip.

Inference feeds `ImpliedVolSurface::from_flat_iv_grid` in
`stochastic-rs-quant`, so a surrogate is a drop-in for the analytic surface.

## Usage

Enable through the umbrella crate:

```toml
[dependencies]
stochastic-rs = { version = "3.0.0-beta.2", features = ["ai"] }
```

## Part of stochastic-rs

This crate is one of the sub-crates of
[**stochastic-rs**](https://github.com/rust-dd/stochastic-rs). Most users
should depend on the umbrella crate, which re-exports everything:

```toml
[dependencies]
stochastic-rs = "3.0.0-beta.2"
```

Depend on `stochastic-rs-ai` directly only when you want this slice and nothing else.

- Documentation: [stochastic.rust-dd.com](https://stochastic.rust-dd.com)
- API reference: [docs.rs/stochastic-rs-ai](https://docs.rs/stochastic-rs-ai)

## License

MIT
