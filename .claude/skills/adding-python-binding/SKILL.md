---
name: adding-python-binding
description: Step-by-step recipe for exposing a new Rust type to Python via stochastic-rs-py. Quickstart companion to the comprehensive `python-bindings` SKILL — invoke for the "what files do I touch and in what order" view.
---

# Adding Python binding — quickstart

This SKILL is the recipe-style companion to the comprehensive
`python-bindings` SKILL. Use it when you have a working Rust type and
need to know **exactly which files to edit, in what order, and how to
verify**. For the deep mechanics (`IntoF32` shim, `unsendable`
semantics, macro internals, error-translation rules), invoke
`python-bindings`.

## 1. Decide which pipeline applies

| Type kind | Pipeline | Macro / pattern |
|---|---|---|
| Distribution (`SimdXxx`) | Macro | `py_distribution!` in producer crate |
| Stochastic process | Macro | `py_process_1d!` / `py_process_2x1d!` / `py_process_2d!` |
| Pricer / Calibrator / VolSurface | Hand-written | `#[pyclass]` block in `stochastic-rs-quant/src/python.rs` |
| Stats estimator | Hand-written | `#[pyfunction]` or `#[pyclass]` block in `stochastic-rs-quant/src/python.rs` |

Macro pipelines (rows 1-2) are nearly mechanical; hand-written pipelines
(rows 3-4) follow patterns set by the existing 198 wrappers.

## 2. The 5-step recipe

For a new type `Foo`:

### Step 1 — wrapper definition

For **macros**, append at the bottom of the producer-crate source file
(e.g. `stochastic-rs-distributions/src/foo.rs`):

```rust
py_distribution!(PyFoo, SimdFoo,
    sig: (a, b, seed = None, dtype = None),
    params: (a: f64, b: f64),
);
```

For **hand-written**, append in `stochastic-rs-quant/src/python.rs`:

```rust
#[pyclass(name = "Foo", unsendable)]
pub struct PyFoo {
    inner: crate::pricing::foo::Foo,
}

#[pymethods]
impl PyFoo {
    #[new]
    #[pyo3(signature = (a, b, c=0.0))]
    fn new(a: f64, b: f64, c: f64) -> PyResult<Self> {
        if a <= 0.0 { return Err(PyValueError::new_err("a must be > 0")); }
        Ok(Self { inner: crate::pricing::foo::Foo { a, b, c } })
    }

    fn price(&self) -> f64 {
        self.inner.calculate_price()
    }
}
```

For **clone-able value types** that may be passed *back* from Python
into another Rust function, add `from_py_object`:

```rust
#[pyclass(name = "FooParams", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyFooParams {
    pub inner: crate::calibration::foo::FooParams,
}
```

### Step 2 — register in `stochastic-rs-py/src/lib.rs`

```rust
use stochastic_rs_quant::python::PyFoo;     // or stochastic_rs_distributions::foo::PyFoo
// ...
m.add_class::<PyFoo>()?;
```

For **`#[pyfunction]`** rather than `#[pyclass]`:

```rust
m.add_function(pyo3::wrap_pyfunction!(
    stochastic_rs_quant::python::my_pyfunction,
    m
)?)?;
```

The use statements at the top are alphabetical; insert in order.
Skipping this step is the #1 cause of `AttributeError: module
'stochastic_rs' has no attribute 'Foo'` after a `maturin develop`.

### Step 3 — compile-check

```bash
cargo check -p stochastic-rs-py
```

If this passes, the wrapper is structurally correct. PyO3 macros emit
useful error messages when the wrapper diverges from the protocol.

### Step 4 — `maturin develop`

```bash
cd stochastic-rs-py
maturin develop --release
```

Drop `--release` for faster iteration during dev. Final wheels for
PyPI are always release.

### Step 5 — smoke test

```python
import stochastic_rs as sr
foo = sr.Foo(a=1.0, b=2.0)
print(foo.price())
```

Or, embedded:

```rust
Python::with_gil(|py| {
    let module = py.import_bound("stochastic_rs")?;
    let foo = module.getattr("Foo")?.call1((1.0, 2.0))?;
    let price: f64 = foo.call_method0("price")?.extract()?;
    assert!(price > 0.0);
    Ok::<_, pyo3::PyErr>(())
}).unwrap();
```

## 3. Common patterns

### Returning numpy arrays

```rust
fn paths<'py>(&self, py: Python<'py>, n: usize) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    let paths: ndarray::Array2<f64> = self.inner.simulate_paths(n);
    paths.into_pyarray(py)
}
```

### Accepting numpy arrays

```rust
fn fit<'py>(&mut self, returns: numpy::PyReadonlyArray1<'py, f64>) -> PyResult<()> {
    self.inner.fit(returns.as_array())
        .map_err(|e| PyValueError::new_err(format!("fit failed: {e}")))
}
```

### Translating Result<T, E> to PyErr

```rust
fn calibrate(&self) -> PyResult<(f64, f64)> {
    self.inner.calibrate(None)
        .map(|res| (res.a, res.b))
        .map_err(|e| PyValueError::new_err(format!("calibration failed: {e}")))
}
```

### Optional argument with default in pyo3 signature

```rust
#[pyo3(signature = (returns, alpha=0.05))]
fn cvar(&self, returns: numpy::PyReadonlyArray1<f64>, alpha: f64) -> PyResult<f64> {
    // ...
}
```

The `=` in the `signature` attribute provides the Python default.
**Do not** use `Option<T>` with `unwrap_or` for defaultable scalars —
the signature attribute is more idiomatic and shows up in `help(...)`.

## 4. Common errors (and fixes)

| Error | Cause | Fix |
|---|---|---|
| `module has no attribute Foo` | Forgot `m.add_class::<PyFoo>()?` | Add to `stochastic-rs-py/src/lib.rs` |
| `the trait FromPyObject is not implemented for PyFoo` | Need to receive PyFoo as fn arg | Add `from_py_object` to the `pyclass` attr |
| Compile error from pyo3 macro | Field has non-PyO3 type | Use `inner: <RustType>` field, expose via methods |
| `AttributeError` after rebuild | `maturin develop` ran in wrong dir | `cd stochastic-rs-py && maturin develop` |
| Segfault on calling method | `#[pyclass]` is missing `unsendable` for non-Send field | Add `unsendable` |

## 5. Checklist before moving on

- [ ] `cargo check -p stochastic-rs-py` passes.
- [ ] `m.add_class::<PyFoo>()?` (or `m.add_function`) registered.
- [ ] `maturin develop` ran; smoke test imports succeed.
- [ ] If method returns `Result`, error path tested in Python with `pytest.raises`.
- [ ] If method accepts numpy, tested with both `np.array(..., dtype=float64)` and `np.float32` (latter should error or convert).

## 6. Where to look when stuck

- The existing 210 wrappers in `stochastic-rs-quant/src/python.rs`
  (most pattern-rich source).
- The `python-bindings` SKILL has the comprehensive mechanics.
- `stochastic-rs-distributions/src/macros.rs` for the
  `py_distribution!` expansion.
- `stochastic-rs-stochastic/src/macros.rs` for the `py_process_*!`
  expansions.

## Related SKILLs

- `python-bindings` — comprehensive reference (this is the quickstart).
- `release-checklist` — how the wrapper ends up in a PyPI wheel.
- `feature-flag-management` — the `#![cfg(feature = "python")]` gate
  on `python.rs`.
