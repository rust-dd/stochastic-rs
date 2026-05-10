---
name: vol-surrogate-nn
description: How to add a neural-network volatility surrogate to stochastic-rs-ai. Covers StochVolModelSpec, BoundedScaler / StandardScaler conventions, gzip-npy training-set loading, train_save_load roundtrip test, and predict_surface integration with ImpliedVolSurface::from_flat_iv_grid.
---

# Vol surrogate NN — stochastic-rs-ai

`stochastic-rs-ai` hosts neural-network surrogates for stochastic-vol
models — calibration-time replacements that replace expensive Heston /
Bates / rBergomi pricers with sub-microsecond MLP forward passes. The
crate is rc.1-experimental and feature-gated upstream.

This SKILL documents the contract for adding a new surrogate (e.g.
SABR, fBates, jumps-on-Heston) so existing tooling (calibration
pipelines, vol-surface pipelines, Python bindings) consumes it without
custom glue.

## 1. The `StochVolModelSpec` contract

```rust
// stochastic-rs-ai/src/spec.rs

pub struct StochVolModelSpec {
    /// Model name: "Heston", "Bates", "rBergomi", ...
    pub name: &'static str,
    /// Names of model parameters in fixed order. Must match the
    /// training-set parameter columns.
    pub param_names: &'static [&'static str],
    /// Bounds for each parameter: lo / hi. Used by BoundedScaler.
    pub param_bounds: &'static [(f64, f64)],
    /// Strike / log-moneyness grid the surrogate predicts on.
    pub k_grid: Vec<f64>,
    /// Maturity grid the surrogate predicts on.
    pub t_grid: Vec<f64>,
}

impl StochVolModelSpec {
    /// Validates that bounds match param_names length, k_grid is
    /// monotonic, t_grid is monotonic + positive. Panics with an
    /// informative message on failure (used at construction time, not
    /// per-inference).
    pub fn new(...) -> Self {
        // assert!(param_names.len() == param_bounds.len(), ...);
        // assert!(k_grid.windows(2).all(|w| w[0] < w[1]), ...);
        // ...
    }
}
```

Construction-time validation is non-negotiable: a surrogate trained
with `param_bounds = [(-1, 1), ...]` and used with `BoundedScaler`
that assumes those bounds will silently produce out-of-distribution
predictions if the bounds drift.

## 2. Scaler conventions

Two scalers, mandatory:

- **`BoundedScaler`**: maps `[lo, hi] → [0, 1]` for *bounded*
  parameters (correlation ρ, Hurst H). Forward: `(x - lo) / (hi - lo)`.
  Inverse: `lo + y * (hi - lo)`.
- **`StandardScaler`**: maps `(x - mu) / sigma` for unbounded
  parameters (log-vol, log-vov). Used when the parameter has tails
  (e.g. log-σ_v under Heston extends to ~ ±5 in practice).

The training set defines `(mu, sigma)` for the StandardScaler — these
are saved alongside the network weights so inference reproduces
training-time normalisation exactly.

## 3. Training-set format: gzip-npy

Training sets are stored as `np.savez_compressed`-compatible `.npz`
files (gzipped npy archives), one per spec:

```
training_data/
  heston.npz       # contains: params (N, 5), iv_grid (N, K, T), spec.json
  bates.npz
  ...
```

Loading via `crate::loader::npz_loader::load_training_set` returns:

```rust
pub struct TrainingSet {
    pub params: Array2<f64>,    // (N, n_params)
    pub iv_grid: Array3<f64>,   // (N, K_grid_len, T_grid_len)
    pub spec: StochVolModelSpec,
}
```

The loader ensures column order in `params` matches `spec.param_names`;
mismatch causes a `LoaderError::ColumnOrderMismatch` rather than
silent wrong-axis indexing.

## 4. Network architecture

A typical surrogate is a small MLP:

```rust
let model = StochVolMLP::new(
    /* input_dim:  */ spec.param_names.len(),
    /* hidden:     */ &[64, 64, 64],
    /* output_dim: */ spec.k_grid.len() * spec.t_grid.len(),
    /* activation: */ Activation::Gelu,
);
```

Output layer is **flat** (K * T scalars). The `predict_surface` step
reshapes to `(K, T)` and feeds `ImpliedVolSurface::from_flat_iv_grid`
for downstream consumers.

## 5. The `predict_surface` contract

```rust
impl<M: StochVolModel> M {
    /// Predict an implied-vol surface for a single parameter set.
    /// Returns an `ImpliedVolSurface` aligned to the spec's k_grid /
    /// t_grid.
    pub fn predict_surface(&self, params: &[f64]) -> ImpliedVolSurface {
        // 1. Validate params.len() == spec.param_names.len()
        // 2. Apply scalers (BoundedScaler / StandardScaler per param)
        // 3. Forward pass through the MLP (output is flat K*T)
        // 4. Reshape + un-scale (output StandardScaler if applicable)
        let flat: Vec<f32> = forward_pass(...);
        ImpliedVolSurface::from_flat_iv_grid(
            self.spec.k_grid.clone(),
            self.spec.t_grid.clone(),
            forwards_for_grid(&self.spec, params),
            &flat,
        )
    }
}
```

Cross-check: `ImpliedVolSurface::from_flat_iv_grid(strikes, maturities,
forwards, flat_ivs)` expects `flat_ivs.len() == N_T * N_K` (note the
order: outer T, inner K). The surrogate's output flattening **must**
match this convention; a transpose bug here silently rotates the
surface 90° and the calibrator fits to the wrong vol.

## 6. Mandatory test: train-save-load roundtrip

```rust
#[test]
fn train_save_load_roundtrip() {
    let spec = StochVolModelSpec::heston_default();
    let trainset = generate_synthetic_trainset(&spec, n_samples = 1_000);
    let model = StochVolMLP::train(&trainset, n_epochs = 10);

    let tmpdir = tempfile::tempdir().unwrap();
    let path = tmpdir.path().join("heston_surrogate.bin");
    model.save(&path).unwrap();
    let loaded = StochVolMLP::load(&path).unwrap();

    // Inference equality on a fixed parameter set:
    let params = vec![0.04, 2.0, 0.04, 0.3, -0.7];
    let surf_orig = model.predict_surface(&params);
    let surf_load = loaded.predict_surface(&params);
    assert!(
        (surf_orig.ivs - surf_load.ivs).iter().all(|d| d.abs() < 1e-6),
        "save/load roundtrip mismatch"
    );
}
```

The roundtrip test catches:
- Missing scaler params in the save format.
- Floating-point drift between `f64` (Rust) and `f32` (model weights).
- Mismatched parameter / activation hyperparameters.

## 7. Real-trainset fit plot (acceptance)

After training, generate a 5×5 grid of (parameter set, surface)
plots comparing:
- Training-target IV (from the slow Rust pricer).
- Surrogate prediction.

The plot should show match within ~10 bps in IV across the grid.
Use `stochastic-rs-viz` for the plotter.

If the fit is visibly off — e.g. the wing tails diverge — the
training set was too small (try 50_000 samples) or the network too
shallow (try 4 hidden layers).

## 8. Anti-patterns

- **Do not** skip the `StochVolModelSpec::new` validation. Bound
  drift between training and inference is the silent killer of
  surrogates.
- **Do not** mix `f32` and `f64` between training-data load and
  inference. Pick one (`f32` is conventional for MLPs) and stick.
- **Do not** flatten the surface in K-major order if
  `from_flat_iv_grid` expects T-major order. Verify with a transpose
  test on a known-asymmetric grid.
- **Do not** ship a surrogate without the train-save-load test.

## 9. Reference impls

- `heston_surrogate.rs` — first surrogate; sets the spec / scaler
  pattern.
- `bates_surrogate.rs` — extends to jumps; same shape.
- `rbergomi_surrogate.rs` — fractional with H as a bounded parameter
  (BoundedScaler).

## Related SKILLs

- `add-fractional-process` — for the underlying data-generation
  process.
- `python-bindings` — exposing the surrogate to Python (deferred to
  v2.x; AI bindings not in v2.0).
- `feature-flag-management` — `--features ai` propagation.
