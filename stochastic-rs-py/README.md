# stochastic-rs-py

**Python (pyo3) bindings for stochastic-rs**

The `stochastic_rs` Python extension module: 234 entries (218 PyO3 classes
and 16 functions) spanning distributions, processes, copulas, statistics
and the quant layer. Numpy in, numpy out.

This crate is a `cdylib` and is **not published to crates.io** — it ships to
PyPI as wheels.

## Install

```bash
pip install stochastic-rs
```

## Usage

```python
import stochastic_rs as srs

p = srs.Ou(theta=2.0, mu=0.0, sigma=1.0, n=1000, x0=0.0, t=1.0)
path = p.sample()                      # numpy.ndarray, shape (1000,)

pricer = srs.HestonPricer(
    s=100, v0=0.04, k=100, r=0.03, kappa=2.0, theta=0.04, sigma=0.3,
    rho=-0.5, tau=1.0, q=0.0,
)
call, put = pricer.call_put()
print(call)
```

## Building from source

```bash
pip install maturin
maturin develop --release --manifest-path stochastic-rs-py/Cargo.toml
```

Linux and macOS wheels ship with the `openblas` feature enabled; the
Windows wheel omits the 13 BLAS-backed classes.

## Part of stochastic-rs

This crate is one of the sub-crates of
[**stochastic-rs**](https://github.com/rust-dd/stochastic-rs). Most users
should depend on the umbrella crate, which re-exports everything:

```toml
[dependencies]
stochastic-rs = "3.0.0-beta.2"
```

Depend on `stochastic-rs-py` directly only when you want this slice and nothing else.

- Documentation: [stochastic.rust-dd.com](https://stochastic.rust-dd.com)
- API reference: [docs.rs/stochastic-rs-py](https://docs.rs/stochastic-rs-py)

## License

MIT
