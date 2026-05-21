#[cfg(feature = "openblas")]
use numpy::PyReadonlyArray1;
#[cfg(feature = "openblas")]
use pyo3::prelude::*;

#[cfg(feature = "openblas")]
#[pyclass(name = "GaussianHmm", unsendable)]
pub struct PyGaussianHmm {
  inner: crate::econometrics::hmm::GaussianHmm,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyGaussianHmm {
  /// Construct a Gaussian-emission HMM with `K` hidden states.
  #[new]
  fn new<'py>(
    initial: PyReadonlyArray1<'py, f64>,
    transitions: numpy::PyReadonlyArray2<'py, f64>,
    means: PyReadonlyArray1<'py, f64>,
    stds: PyReadonlyArray1<'py, f64>,
  ) -> Self {
    Self {
      inner: crate::econometrics::hmm::GaussianHmm::new(
        initial.as_array().to_owned(),
        transitions.as_array().to_owned(),
        means.as_array().to_owned(),
        stds.as_array().to_owned(),
      ),
    }
  }

  fn n_states(&self) -> usize {
    self.inner.n_states()
  }

  fn log_likelihood<'py>(&self, observations: PyReadonlyArray1<'py, f64>) -> f64 {
    self.inner.log_likelihood(observations.as_array())
  }

  fn viterbi<'py>(
    &self,
    py: Python<'py>,
    observations: PyReadonlyArray1<'py, f64>,
  ) -> pyo3::Bound<'py, numpy::PyArray1<usize>> {
    use numpy::IntoPyArray;
    self.inner.viterbi(observations.as_array()).into_pyarray(py)
  }

  /// Train via Baum-Welch EM and return `(iterations, log_likelihood, converged)`.
  fn baum_welch<'py>(
    &mut self,
    observations: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
  ) -> (usize, f64, bool) {
    let fit = self
      .inner
      .baum_welch(observations.as_array(), max_iter, tol);
    (fit.iterations, fit.log_likelihood, fit.converged)
  }
}
