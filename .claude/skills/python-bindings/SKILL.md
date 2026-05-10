---
name: python-bindings
description: Maintenance and extension rules for stochastic-rs Python (PyO3) bindings — invoke when adding/updating distributions, processes, pricers, calibrators, or estimators that need Python exposure
---

# Python Bindings (PyO3) — stochastic-rs

The `stochastic-rs-py` cdylib re-exports `#[pyclass]`-wrapped types from
producer crates (`stochastic-rs-distributions`, `-stochastic`, `-quant`,
`-stats`, `-copulas`). At rc.1 the surface is **210 entries: 198 PyO3
classes + 12 pyfunctions** across distributions / stochastic / quant /
copulas / stats. AI bindings deferred to 2.x.

This skill covers four pipelines:

1. **Distributions** — wrapped via `py_distribution!` / `py_distribution_int!`
2. **Stochastic processes** — wrapped via `py_process_1d!` / `py_process_2x1d!` / `py_process_2d!`
3. **Quant pricers / calibrators / vol surfaces** — hand-written
   `#[pyclass]` blocks in `stochastic-rs-quant/src/python.rs`
4. **Stats estimators** — hand-written `#[pyfunction]`/`#[pyclass]` blocks

For all four, the entry point is `stochastic-rs-py/src/lib.rs`, which
imports each wrapper and registers it via `m.add_class::<PyXxx>()` /
`m.add_function(wrap_pyfunction!(py_xxx, m)?)`.

## 1. Macros (auto-generated wrappers)

### 1.1. `py_distribution!` — for `Simd*` distribution types

Producer crate: `stochastic-rs-distributions`. Macro definition in
`stochastic-rs-distributions/src/macros.rs`. Invoke at the **bottom of the
distribution's source file**, e.g. `stochastic-rs-distributions/src/normal.rs`:

```rust
py_distribution!(PyNormal, SimdNormal,
  sig: (mean, std, seed = None, dtype = None),
  params: (mean: f64, std: f64),
);
```

What you get:
- `PyNormal` `#[pyclass(unsendable)]`
- `__new__(mean, std, seed=None, dtype=None)` — `seed: Option<u64>`,
  `dtype: Option<&str>` ∈ {"f32", "f64"}, default f64
- `sample(n)` returning `numpy.ndarray`
- `sample_par(m, n)` returning `numpy.ndarray` (parallel matrix)

`IntoF32` / `IntoF64` shims in `stochastic-rs-core::python` convert each
parameter to the chosen dtype. **All distribution parameters MUST be
`f64`-typed in the macro `params:` clause** — the dispatching to
`SimdXxx<f32>` happens via the shim.

### 1.2. `py_distribution_int!` — for integer-valued distributions

Same shape but for `Poisson`, `Binomial`, `Geometric`, etc. that produce
`u32`/`i32` arrays.

### 1.3. `py_process_1d!` / `py_process_2x1d!` / `py_process_2d!`

Producer crate: `stochastic-rs-stochastic`. Macro definition in
`stochastic-rs-stochastic/src/macros.rs`. Pick by the process's output
shape:

| Macro | Output shape | Examples |
|---|---|---|
| `py_process_1d!` | `Array1<T>` (single 1-D path) | GBM, OU, Vasicek, fBM, Cir |
| `py_process_2x1d!` | `[Array1<T>; 2]` (two 1-D paths) | Heston (price, vol), 2-D Brownian |
| `py_process_2d!` | `Array2<T>` (correlated multi-asset) | Multi-asset GBM |

Invoke at the **bottom of the process's source file**, e.g.
`stochastic-rs-stochastic/src/diffusion/gbm.rs`:

```rust
py_process_1d!(PyGBM, GBM,
  sig: (mu, sigma, n, x0 = 1.0, t = 1.0, m = None, seed = None),
  params: (mu: f64, sigma: f64, n: usize, x0: f64, t: f64, m: Option<usize>),
);
```

You get:
- `PyGBM.sample()` returning a numpy array
- `PyGBM.sample_seeded(seed)` for deterministic reproduction
- Internal `f64` only (no f32 dispatch for processes — too much code-gen
  growth for marginal benefit)

## 2. Hand-written `#[pyclass]` (quant pricers / calibrators)

Located in `stochastic-rs-quant/src/python.rs`. The module is feature-gated
behind `#[cfg(feature = "python")]` (whole file). Pattern:

```rust
#[pyclass(name = "BSMPricer", unsendable)]
pub struct PyBSMPricer {
  inner: crate::pricing::bsm::BSMPricer,
}

#[pymethods]
impl PyBSMPricer {
  #[new]
  #[pyo3(signature = (s, k, r, q, sigma, t))]
  fn new(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> PyResult<Self> {
    if sigma <= 0.0 {
      return Err(PyValueError::new_err("sigma must be > 0"));
    }
    Ok(Self {
      inner: crate::pricing::bsm::BSMPricer { s, k, r, q, sigma, t,
        eval: None, expiration: None },
    })
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
}
```

### 2.1. Calibrators (Result-typed)

When the inner Rust `Calibrator` returns `Result`, propagate into Python
via `PyResult` and `PyValueError::new_err`:

```rust
fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, bool)> {
  use crate::traits::Calibrator;
  let res = self.inner.calibrate(None)
    .map_err(|e| PyValueError::new_err(format!("calibration failed: {e}")))?;
  Ok((res.params().alpha, res.params().beta, res.params().nu,
      res.params().rho, res.converged()))
}
```

This is the **standard pattern** for the unified `Calibrator` trait.
Cgmysv / HKDE / Hscm / RBergomi calibrators all follow it.

### 2.2. Constructors that validate user input

Returning `PyResult<Self>` from `#[new]` is fully supported by PyO3 and
preferred over `panic!`. Example pattern:

```rust
#[new]
fn new(alpha: f64, beta: f64, delta: f64, r: f64, q: f64) -> PyResult<Self> {
  if alpha <= 0.0 {
    return Err(PyValueError::new_err("alpha must be > 0"));
  }
  if !(beta.abs() < alpha) {
    return Err(PyValueError::new_err("|beta| must be < alpha"));
  }
  Ok(Self { ... })
}
```

## 3. Registration in `stochastic-rs-py/src/lib.rs`

Every wrapped type **must** be registered explicitly. The cdylib's
`#[pymodule]` `init` function:

```rust
#[pymodule]
fn stochastic_rs(_py: Python, m: &PyModule) -> PyResult<()> {
  // 1. Imports — alphabetised within each producer crate's section.
  use stochastic_rs_quant::python::PyBSMPricer;
  use stochastic_rs_quant::python::PyNigFourier;     // new in rc.2
  // ...

  // 2. Registration — group by domain (pricer, calibrator, vol_surface, ...).
  m.add_class::<PyBSMPricer>()?;
  m.add_class::<PyNigFourier>()?;
  // ...

  Ok(())
}
```

**Forgotten `add_class` is the #1 cause of "module has no attribute X" at
import time.** Always register a new wrapper in the same PR that adds the
`#[pyclass]`.

## 4. Build & test workflow

### 4.1. Local development

```bash
# From the workspace root.
cd stochastic-rs-py

# Activate / create a venv (only first time):
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy

# Build + install in dev mode (~30s):
maturin develop --release

# Smoke test:
python -c "from stochastic_rs import PyBSMPricer; \
  p = PyBSMPricer(100, 100, 0.05, 0.0, 0.2, 1.0); print(p.price())"
```

`pyproject.toml` has `[tool.maturin] manifest-path = ".../Cargo.toml"`
pointing at the cdylib crate.

### 4.2. Smoke testing

For every new `#[pyclass]` add a `tests/python/test_<topic>.py` (driven
from the cdylib crate) that exercises:

- `__new__` happy path
- `__new__` invalid-input branch (expect `ValueError`)
- The 1–2 most-used methods (`sample`, `price`, `calibrate`, …)

Or write the smoke test inline in Rust under `#[cfg(test)]` using
`pyo3::Python::with_gil(|py| { ... })` — that lets `cargo test --features
python` cover it without spawning Python.

### 4.3. CI

`cargo test --workspace --exclude stochastic-rs-py` covers the Rust side.
Python smoke is on the maturin path under
`.github/workflows/ci.yml::wheels`.

## 5. Common patterns / gotchas

### 5.1. Sendability

PyO3 `#[pyclass]` wrappers default to `Send + Sync`. For `unsendable` types
(those holding `Rc<RefCell<...>>`, `RefCell<...>`, or non-`Send` external
handles like `YahooConnector`), add `unsendable`:

```rust
#[pyclass(name = "RBergomiCalibrator", unsendable)]
```

Forgetting `unsendable` on a non-`Send` type triggers a compile error
("the trait `Send` is not implemented for `Rc<...>`").

### 5.2. Numpy interop

For `Array1<f64>` / `Array2<f64>` returns, use:

```rust
use numpy::IntoPyArray;
fn surface(&self, py: Python) -> Py<PyAny> {
  self.inner.surface().into_pyarray(py).into_py_any(py).unwrap()
}
```

For inputs, accept `PyReadonlyArray1<f64>` / `PyReadonlyArray2<f64>` and
call `.as_array()` to get an `ArrayView`.

### 5.3. Optional parameters with defaults

`#[pyo3(signature = (...))]` controls Python-visible defaults. Required:
list every parameter, even if you didn't change it. Example with
optional + default:

```rust
#[pyo3(signature = (s0, r, slices, hurst=0.1, rho=-0.7, eta=2.0, xi0=0.04,
                    max_iters=60, paths=1024))]
fn new(...) -> PyResult<Self> { ... }
```

### 5.4. Return-tuple naming

Calibrators that produce many scalars typically return tuples. Document
the order in the calibrator's docstring:

```rust
/// Calibrate. Returns `(alpha, beta, nu, rho, converged, rmse)`.
fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, bool, f64)> { ... }
```

For more than ~5 outputs, return a `#[pyclass]` result struct instead.

### 5.5. NaN propagation (rc.1+)

`CarrMadanPricer::price_call` returns `f64::NAN` for out-of-grid strikes
(was `0.0` in rc.0). Python users **must** detect via `math.isnan()`.
Document this on the `price_*_call` methods and expose `strike_in_grid_*`
helpers so callers can pre-check.

## 6. Distribution-specific testing checklist

When wrapping a distribution, confirm:

- [ ] `py_distribution!` invocation at end of source file
- [ ] `from_seed_source` path uses `Deterministic::new(seed)`
- [ ] `IntoF32` / `IntoF64` shims wrap every numeric parameter
- [ ] `m.add_class::<PyXxx>()` added to `stochastic-rs-py/src/lib.rs`
- [ ] Python smoke test (`from stochastic_rs import PyXxx; PyXxx(...)`)
- [ ] `sample(n)` returns the right shape
- [ ] `sample_par(m, n)` returns shape `(m, n)`

## 7. Process-specific testing checklist

- [ ] Choose macro: `py_process_1d!` (1-D) / `py_process_2x1d!` (2 × 1-D) /
      `py_process_2d!` (multi-asset)
- [ ] Macro invocation at end of source file
- [ ] `m.add_class::<PyXxx>()` added to `stochastic-rs-py/src/lib.rs`
- [ ] `PyXxx().sample()` smoke test
- [ ] `PyXxx().sample_seeded(42)` reproduces — exact bit equality

## 8. Pricer / calibrator testing checklist

- [ ] `#[pyclass(name = "...", unsendable?)]` matches the pyi name
- [ ] `#[pyo3(signature = (...))]` lists every parameter
- [ ] Constructors validate inputs and return `PyResult<Self>` with
      `PyValueError::new_err`
- [ ] Calibrators map inner `anyhow::Error` to `PyValueError`
- [ ] `m.add_class::<PyXxx>()` registered
- [ ] Smoke test: construct + 1-2 method calls
- [ ] Round-trip test where applicable (e.g. price → IV → price)

## 9. PyPI release (out of scope here — see release-checklist SKILL)

The `release-checklist` SKILL covers:
- Workspace version bump on 9 crates
- `maturin build --release` per platform (Linux/macOS x86_64, macOS aarch64,
  Windows x86_64)
- `twine upload`
- pyi stub regeneration (`maturin generate-stubs` if used)

This SKILL only covers the wrapper and registration steps.

## 10. References

- PyO3 user guide: <https://pyo3.rs>
- maturin manual: <https://www.maturin.rs>
- numpy crate: <https://docs.rs/numpy>
- Module entry point: `stochastic-rs-py/src/lib.rs`
- Macro definitions: `stochastic-rs-distributions/src/macros.rs`,
  `stochastic-rs-stochastic/src/macros.rs`
- Quant hand-written wrappers: `stochastic-rs-quant/src/python.rs`
