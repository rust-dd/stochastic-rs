[![Crates.io](https://img.shields.io/crates/v/stochastic-rs-viz?style=flat-square)](https://crates.io/crates/stochastic-rs-viz)
[![docs.rs](https://img.shields.io/docsrs/stochastic-rs-viz?style=flat-square)](https://docs.rs/stochastic-rs-viz)
![License](https://img.shields.io/crates/l/stochastic-rs-viz?style=flat-square)

# stochastic-rs-viz

**Plotly-based visualization for stochastic-rs**

A thin plotting layer over [`plotly`](https://crates.io/crates/plotly) for
looking at paths, distributions and surfaces without hand-rolling a chart.

## What is in it

- A grid plotter for laying out several processes or distributions side by
  side in one HTML page.
- Category helpers for diffusion, jump, noise and process families.
- Density-versus-analytic overlays for checking a sampler against its
  closed-form pdf.

## Usage

```rust
use stochastic_rs_viz::{GridPlotter, Plottable, plot_process};
```

The convenience helpers are `plot_process`, `plot_distribution`,
`plot_heatmap` and `plot_vol_surface`; `GridPlotter` composes several of
them into one page, and `Plottable` is the trait a type implements to
become plottable.

Output is a self-contained HTML file you open in a browser.

## Part of stochastic-rs

This crate is one of the sub-crates of
[**stochastic-rs**](https://github.com/rust-dd/stochastic-rs). Most users
should depend on the umbrella crate, which re-exports everything:

```toml
[dependencies]
stochastic-rs = "2.6"
```

Depend on `stochastic-rs-viz` directly only when you want this slice and nothing else.

- Documentation: [stochastic.rust-dd.com](https://stochastic.rust-dd.com)
- API reference: [docs.rs/stochastic-rs-viz](https://docs.rs/stochastic-rs-viz)

## License

MIT
