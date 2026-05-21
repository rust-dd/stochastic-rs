use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Random-walk Metropolis-Hastings with a Python-callable log target.
///
/// `log_target` is a Python function `numpy.ndarray -> float` that may return
/// `-inf` to encode hard constraints. The chain runs in pure Python-callback
/// mode (single-threaded, GIL-bound), so it is most useful for low-dimensional
/// parameter spaces. For performance-critical work consider porting the
/// log-density to Rust and exposing it as a closed-form distribution instead.
#[pyfunction]
#[pyo3(signature = (initial, log_target, proposal_scale, n_samples, burn_in=1000, seed=42))]
pub fn random_walk_metropolis<'py>(
  py: Python<'py>,
  initial: PyReadonlyArray1<'py, f64>,
  log_target: pyo3::Py<pyo3::PyAny>,
  proposal_scale: PyReadonlyArray1<'py, f64>,
  n_samples: usize,
  burn_in: usize,
  seed: u64,
) -> PyResult<(
  pyo3::Bound<'py, numpy::PyArray2<f64>>,
  pyo3::Bound<'py, numpy::PyArray1<f64>>,
  f64,
)> {
  use ndarray::ArrayView1;
  use numpy::IntoPyArray;
  let init_arr = initial.as_array().to_owned();
  let scale_arr = proposal_scale.as_array().to_owned();
  let log_target_ref = &log_target;
  let target_fn = |theta: ArrayView1<f64>| -> f64 {
    let theta_owned: Vec<f64> = theta.iter().copied().collect();
    let theta_pyarr = numpy::PyArray1::from_vec(py, theta_owned);
    let res = log_target_ref.call1(py, (theta_pyarr,));
    match res {
      Ok(obj) => obj.extract::<f64>(py).unwrap_or(f64::NEG_INFINITY),
      Err(_) => f64::NEG_INFINITY,
    }
  };
  let res = crate::filtering::mcmc::random_walk_metropolis(
    init_arr.view(),
    target_fn,
    scale_arr.view(),
    n_samples,
    burn_in,
    seed,
  );
  Ok((
    res.samples.into_pyarray(py),
    res.log_targets.into_pyarray(py),
    res.acceptance_rate,
  ))
}
