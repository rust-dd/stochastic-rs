---
name: vol-surrogate-nn
description: How to add a neural-network volatility surrogate to stochastic-rs-ai. Covers StochVolModelSpec, StochVolNn, the BoundedScaler / StandardScaler split, gzip-npy training-set loading, the train_save_load roundtrip test, and predict_implied_vol_surface integration with ImpliedVolSurface.
---

# Vol surrogate NN — stochastic-rs-ai

`stochastic-rs-ai` hosts neural-network surrogates for stochastic-vol
models — calibration-time replacements for expensive Heston / rBergomi
pricers. The crate is feature-gated upstream (`--features ai` on the
umbrella).

The whole crate is small and worth reading end to end before adding a
surrogate: `stochastic-rs-ai/src/volatility/` is four reusable modules
under `common/` plus **three** thin model wrappers (`heston.rs`,
`one_factor.rs`, `rbergomi.rs`). A new surrogate is a fourth wrapper —
you almost certainly write no new network code at all.

## 1. Architecture: one shared engine, thin per-model wrappers

```
volatility/common/
  spec.rs      StochVolModelSpec, TrainConfig, TrainReport, EpochMetrics
  model.rs     StochVolNn            <- the engine; everything goes through it
  network.rs   FeedForwardNet        <- pub(super); you do not touch this
  scaler.rs    BoundedScaler, StandardScaler   <- pub(super)
  dataset.rs   load_trainset_gzip_npy, rmse_1d
  metadata.rs  save/load metadata serialisation
  plot.rs      write_surface_fit_plot_html     <- #[cfg(feature = "viz")]
volatility/heston.rs | one_factor.rs | rbergomi.rs   <- the wrappers
```

`FeedForwardNet` is `pub(super)` and fixed: **3 hidden ELU layers of
uniform width `hidden_dim`, plus a linear output layer**. There is no
per-layer width vector and no activation parameter. If your model needs
a different architecture, that is a change to `network.rs`, not a
parameter you pass.

There is no `StochVolMLP` type. The engine is `StochVolNn`.

## 2. The `StochVolModelSpec` contract

```rust
// stochastic-rs-ai/src/volatility/common/spec.rs

#[derive(Clone, Debug)]
pub struct StochVolModelSpec {
  pub model_id: String,      // "heston", "rbergomi", … — checked on load
  pub input_dim: usize,      // number of model parameters
  pub output_dim: usize,     // flat surface length = n_maturities × n_strikes
  pub hidden_dim: usize,     // uniform width of all 3 hidden layers
  pub param_lb: Vec<f32>,    // per-parameter lower bounds, len == input_dim
  pub param_ub: Vec<f32>,    // per-parameter upper bounds, len == input_dim
}

impl StochVolModelSpec {
  pub fn new(
    model_id: impl Into<String>,
    input_dim: usize, output_dim: usize, hidden_dim: usize,
    param_lb: Vec<f32>, param_ub: Vec<f32>,
  ) -> Result<Self>    // anyhow::Result — bails, does NOT panic
}
```

Everything is `f32`, not `f64`. There are no `param_names`, no `k_grid`,
no `t_grid` fields: the spec knows only how many parameters go in and
how many numbers come out. The strike / maturity grid is supplied by the
**caller** at `predict_implied_vol_surface` time, and keeping the two
consistent is the caller's job — the spec cannot check it for you
beyond `output_dim == n_k × n_t`.

`new` returns `Result` and `bail!`s on: any zero dimension, a
`param_lb` / `param_ub` length that does not match `input_dim`, a
non-finite bound, or `param_ub[i] <= param_lb[i]`. Propagate the
`Result`; do not `unwrap` it in a wrapper's constructor.

## 3. The wrapper pattern

Copy `heston.rs`. A wrapper is public constants + a newtype over
`StochVolNn` + four delegating methods:

```rust
pub const MODEL_ID: &str = "heston";
pub const INPUT_DIM: usize = 5;
pub const OUTPUT_DIM: usize = 88;
pub const DEFAULT_HIDDEN_DIM: usize = 30;
pub const PARAM_LB: [f32; INPUT_DIM] = [0.0001, -0.95, 0.01, 0.01, 1.0];
pub const PARAM_UB: [f32; INPUT_DIM] = [0.04, -0.1, 1.0, 0.2, 10.0];

pub struct HestonNn {
  inner: StochVolNn,
}

impl HestonNn {
  pub fn new(device: &Device) -> Result<Self> {
    Self::with_hidden(device, DEFAULT_HIDDEN_DIM)
  }

  pub fn with_hidden(device: &Device, hidden_dim: usize) -> Result<Self> {
    let spec = StochVolModelSpec::new(
      MODEL_ID, INPUT_DIM, OUTPUT_DIM, hidden_dim,
      PARAM_LB.to_vec(), PARAM_UB.to_vec(),
    )?;
    Ok(Self { inner: StochVolNn::new(spec, device)? })
  }

  pub fn train(&mut self, params: &Array2<f32>, surfaces: &Array2<f32>,
               config: &TrainConfig) -> Result<TrainReport> {
    self.inner.train(params, surfaces, config)
  }

  pub fn predict_surface(&self, params: &[f32; INPUT_DIM]) -> Result<Vec<f32>> {
    self.inner.predict_surface(params)
  }

  pub fn save<P: AsRef<Path>>(&self, dir: P) -> Result<()> {
    self.inner.save(dir)
  }

  /// Note the arity: the wrapper supplies `MODEL_ID` itself, so callers
  /// pass only `(dir, device)`. The engine's own `load` takes the
  /// expected id first.
  pub fn load<P: AsRef<Path>>(dir: P, device: &Device) -> Result<Self> {
    Ok(Self { inner: StochVolNn::load(MODEL_ID, dir, device)? })
  }
}
```

Also delegate `predict_surfaces(&Array2<f32>)` and — behind
`#[cfg(feature = "quant")]` — `predict_implied_vol_surface`, both
taking `&[f32; INPUT_DIM]` on the wrapper.

Exporting `INPUT_DIM` / `OUTPUT_DIM` / `PARAM_LB` / `PARAM_UB` as `pub
const` is the convention — downstream code sizes its arrays off them,
and the fixed-size `&[f32; INPUT_DIM]` argument on `predict_surface` is
what turns an arity mistake into a compile error.

Cite the architecture's source in the module `//!` header, as the three
existing wrappers do (they follow
`github.com/amuguruza/NN-StochVol-Calibrations`).

Register the new module in `stochastic-rs-ai/src/volatility.rs`.

## 4. Scaler conventions — both are automatic

`BoundedScaler` and `StandardScaler` are `pub(super)`. You do not
construct them; `StochVolNn` wires them:

- **`BoundedScaler`** scales the **inputs**, from `spec.param_lb /
  param_ub` to `[-1, 1]`. It is built from the spec at `StochVolNn::new`
  — this is what makes the bounds load-bearing rather than
  documentation.
- **`StandardScaler`** scales the **outputs** (the IV surface), fitted
  on the training set (`fit` → `transform` / `inverse_transform`). It is
  `Option`-typed on the model: `predict_surface` errors with
  `"model is not trained or loaded (missing output scaler)"` if you
  predict before training or loading, and `save` errors on an untrained
  model.

The output scaler's fitted mean/std are serialised into the metadata
file next to the weights, so inference reproduces training-time
normalisation exactly.

## 5. Training-set format: a single gzipped `.npy`

Not `.npz`, not a multi-array archive:

```rust
pub fn load_trainset_gzip_npy<P: AsRef<Path>>(
  path: P, input_dim: usize, output_dim: usize, max_rows: Option<usize>,
) -> Result<(Array2<f32>, Array2<f32>)>
```

It gunzips one `Array2<f64>` `.npy`, then **column-slices** it:
columns `0..input_dim` are the parameters, columns
`input_dim..input_dim+output_dim` are the flat surfaces, and both are
cast down to `f32`. Extra trailing columns are ignored; too few columns
`bail!`s.

There is no `TrainingSet` struct, no `spec.json` inside the archive, and
no column-order validation — the loader trusts the column layout. Row
order therefore **is** the contract: if your generator writes parameters
in a different order than `PARAM_LB` / `PARAM_UB`, nothing will tell
you, and the BoundedScaler will silently scale the wrong parameter by
the wrong bounds. Pin the generator's column order in the wrapper's
module doc.

## 6. Prediction and the `ImpliedVolSurface` bridge

Two methods, and the second is feature-gated:

```rust
// always available — flat Vec<f32> of length spec.output_dim
pub fn predict_surface(&self, params: &[f32]) -> Result<Vec<f32>>

// #[cfg(feature = "quant")] on stochastic-rs-ai
pub fn predict_implied_vol_surface(
  &self, params: &[f32],
  strikes: Vec<f64>, maturities: Vec<f64>, forwards: Vec<f64>,
) -> Result<stochastic_rs_quant::vol_surface::ImpliedVolSurface>
```

Also on the engine: `predict_surfaces(&Array2<f32>) -> Result<Array2<f32>>`
for a batch.

**Layout.** The flat prediction is `(N_T, N_K)` row-major —
**maturity-major, strike-minor**. `predict_implied_vol_surface` checks
`forwards.len() == maturities.len()` and `spec.output_dim == n_k × n_t`,
then reshapes. A transpose bug here silently rotates the surface and the
calibrator fits the wrong vol; the shape check catches a *wrong count*,
not a *wrong order*, so verify the order on a deliberately non-square
grid.

The `quant` feature on `stochastic-rs-ai` is a bridge feature
(`quant = ["dep:stochastic-rs-quant"]`) and the umbrella's `ai` feature
turns it on (`ai = ["dep:stochastic-rs-ai", "stochastic-rs-ai/quant"]`).
A surrogate must still compile and be useful with `quant` off — keep
anything `ImpliedVolSurface`-shaped behind the gate.

## 7. Save / load — a **directory**, not a file

```rust
// on the engine, StochVolNn:
pub fn save<P: AsRef<Path>>(&self, dir: P) -> Result<()>
pub fn load<P: AsRef<Path>>(expected_model_id: &str, dir: P, device: &Device) -> Result<Self>

// on a wrapper, e.g. HestonNn — MODEL_ID is supplied internally:
pub fn load<P: AsRef<Path>>(dir: P, device: &Device) -> Result<Self>
```

`save` creates `dir` and writes two files: the candle `VarMap` weights
and a metadata file (spec + fitted output-scaler statistics). The
engine's `load` takes the **expected `model_id` first** and `bail!`s if
the metadata disagrees — that is the guard against loading a Heston
checkpoint into an rBergomi wrapper, and it is why each wrapper hard-codes
its own `MODEL_ID` at the call site instead of exposing the argument.

## 8. Mandatory test: `train_save_load_roundtrip`

All three existing wrappers carry a test of exactly this name; yours
must too. The shape (see `heston.rs`'s test module):

1. Build the model on `Device::Cpu` (`HestonNn::new(&device)?`).
2. Generate a small synthetic set with
   `synthetic_surface_dataset(&PARAM_LB, &PARAM_UB, samples, OUTPUT_DIM, seed)`
   — `pub(crate)`, re-exported only under `#[cfg(test)]`, which is
   exactly what it is for. Heston's test uses `samples = 192, seed = 7`.
3. `train` a handful of epochs with an explicit `TrainConfig` whose
   `random_seed` is pinned, then assert
   `report.epochs.len() == cfg.epochs` and that the final `val_rmse` is
   finite.
4. `save` to a directory under `std::env::temp_dir()` (the three
   existing tests build a per-process name and `remove_dir_all` on both
   ends — there is no `tempfile` dev-dependency here), `load` it back,
   and assert `predict_surface` agrees on a parameter row taken from
   the generated set. Heston's tolerance is `max_diff < 1e-4`.

It catches missing scaler statistics in the metadata, `f32`/`f64` drift,
and mismatched hyperparameters. `TrainConfig::default()` is
`test_ratio 0.15, batch_size 32, epochs 200, learning_rate 1e-3,
random_seed 42, shuffle true` — override `epochs` down for the test
rather than relying on the default.

## 9. Fit plot (optional, `viz`-gated)

`common::write_surface_fit_plot_html` is available behind
`#[cfg(feature = "viz")]` and writes HTML — never `.show()`. Use it to
eyeball training-target IV against surrogate prediction. If the wings
diverge, the training set is too small or the network too narrow
(`hidden_dim` is your only architectural dial).

## 10. Anti-patterns

- **Do not** `unwrap()` `StochVolModelSpec::new`. It returns
  `anyhow::Result` and bails; propagate it.
- **Do not** reach for `f64`. The whole crate is `f32` end to end; the
  loader casts down at the boundary on purpose.
- **Do not** invent spec fields (`param_names`, `k_grid`, `t_grid`).
  The grid travels as arguments to `predict_implied_vol_surface`.
- **Do not** pass a file path to `save` / `load`. They take a directory.
- **Do not** assume strike-major flattening. It is maturity-major.
- **Do not** ship a surrogate without `train_save_load_roundtrip`.
- **Do not** hand-roll a network. Reuse `StochVolNn`; if the
  architecture genuinely must differ, change `network.rs` deliberately
  and say why in its module doc.

## 11. Reference impls

- `heston.rs` — `HestonNn`, 5 params → 88 outputs, hidden 30. The
  template; read its test module.
- `rbergomi.rs` — rough-Bergomi; Hurst as a bounded parameter.
- `one_factor.rs` — one-factor SV; the thinnest of the three.

## Related SKILLs

- `add-fractional-process` — the data-generating process behind an
  rBergomi-class training set.
- `feature-flag-management` — `ai` / `quant` / `viz` propagation.
- `python-bindings` — AI bindings are **not** shipped (deferred past
  2.x); there is no `PyHestonNn`.
