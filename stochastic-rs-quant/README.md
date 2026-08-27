[![Crates.io](https://img.shields.io/crates/v/stochastic-rs-quant?style=flat-square)](https://crates.io/crates/stochastic-rs-quant)
[![docs.rs](https://img.shields.io/docsrs/stochastic-rs-quant?style=flat-square)](https://docs.rs/stochastic-rs-quant)
![License](https://img.shields.io/crates/l/stochastic-rs-quant?style=flat-square)

# stochastic-rs-quant

**Quantitative finance: pricing, calibration, vol surfaces, instruments**

The derivatives layer: pricing engines, model calibration, volatility
surfaces, curves, instruments and risk.

## What is in it

- **Closed-form pricing** — Black-Scholes-Merton, Bachelier, Black-76,
  Bjerksund-Stensland 2002, digitals, barriers, lookbacks, chooser,
  compound, cliquet, Stulz rainbow, Kirk spread, Margrabe.
- **Fourier pricing** — Heston, Bates, Merton jump, Kou, variance gamma,
  CGMY, double Heston, HKDE, CGMYSV, with Carr-Madan and FRFT engines.
- **Numerical pricing** — finite differences, CRR and short-rate lattices
  (Hull-White, Black-Karasinski, G2++), Bermudan LSM, Snell envelope,
  Heston SLV, autocallables, baskets.
- **Greeks** — first and second order from each pricer's inherent
  `greeks(s, k, r, q, tau, option_type)`, plus Malliavin Greeks
  (Thalmaier and El Khatib schemes, which expose `GreeksExt`) and a
  Fourier-Malliavin volatility estimator.
- **Calibration** — Heston (Cui analytic Jacobian), SABR, SVJ, Lévy, rough
  Bergomi, double Heston, HKDE, Hull-White swaption grids, BSM.
- **Volatility surfaces** — implied surfaces from quotes, SVI, SSVI, SABR
  smiles, arbitrage repair, Dupire local vol, Breeden-Litzenberger.
- **Fixed income** — curve bootstrapping, Nelson-Siegel, Svensson,
  multi-curve, bonds, swaps, caps and floors, swaptions, CMS, inflation.
- **Credit** — Merton structural model, hazard-rate bootstrap, CDS, JLT
  migration matrices.
- **Risk and portfolio** — VaR, CVaR, expected shortfall, drawdown,
  Sharpe / Sortino / Calmar, Markowitz, HRP, CVaR optimisation, PCA and
  Fama-MacBeth factors.
- **Microstructure** — Almgren-Chriss, Kyle (1985), Bouchaud propagator and
  a price-time priority order book.

## Usage

```rust
use stochastic_rs_quant::pricing::heston::HestonPricer;

// The struct holds the model; the query travels as arguments, so one
// instance prices a whole strike/maturity grid.
let model = HestonPricer::new(0.04, -0.5, 2.0, 0.04, 0.3, Some(0.0));
let (call, put) = model.call_put(100.0, 100.0, 0.03, 0.0, 1.0);
```

## Part of stochastic-rs

This crate is one of the sub-crates of
[**stochastic-rs**](https://github.com/rust-dd/stochastic-rs). Most users
should depend on the umbrella crate, which re-exports everything:

```toml
[dependencies]
stochastic-rs = "3.0.0-beta.2"
```

Depend on `stochastic-rs-quant` directly only when you want this slice and nothing else.

- Documentation: [stochastic.rust-dd.com](https://stochastic.rust-dd.com)
- API reference: [docs.rs/stochastic-rs-quant](https://docs.rs/stochastic-rs-quant)

## License

MIT
