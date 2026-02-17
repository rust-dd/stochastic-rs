use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::noise::cgns::CGNS;
use crate::stochastic::process::cpoisson::CompoundPoisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Bates1996<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub mu: Option<T>,
  pub b: Option<T>,
  pub r: Option<T>,
  pub r_f: Option<T>,
  pub lambda: T,
  pub k: T,
  pub alpha: T,
  pub beta: T,
  pub sigma: T,
  pub rho: T,
  pub n: usize,
  pub s0: Option<T>,
  pub v0: Option<T>,
  pub t: Option<T>,
  pub use_sym: Option<bool>,
  cgns: CGNS<T>,
  pub cpoisson: CompoundPoisson<T, D>,
}

impl<T, D> Bates1996<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    mu: Option<T>,
    b: Option<T>,
    r: Option<T>,
    r_f: Option<T>,
    lambda: T,
    k: T,
    alpha: T,
    beta: T,
    sigma: T,
    rho: T,
    n: usize,
    s0: Option<T>,
    v0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    cpoisson: CompoundPoisson<T, D>,
  ) -> Self {
    Self {
      mu,
      b,
      r,
      r_f,
      lambda,
      k,
      alpha,
      beta,
      sigma,
      rho,
      n,
      s0,
      v0,
      t,
      use_sym,
      cgns: CGNS::new(rho, n - 1, t),
      cpoisson,
    }
  }

  #[inline]
  fn effective_drift(&self) -> T {
    match (self.r, self.r_f, self.b, self.mu) {
      (Some(r), Some(r_f), _, _) => r - r_f,
      (_, _, Some(b), _) => b,
      (_, _, _, Some(mu)) => mu,
      _ => panic!("one of (r and r_f), b, or mu must be provided"),
    }
  }
}

impl<T, D> ProcessExt<T> for Bates1996<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = &self.cgns.sample();
    let jump_increments = self.cpoisson.sample_grid_increments(self.n, dt);

    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);

    s[0] = self.s0.unwrap_or(T::zero());
    v[0] = self.v0.unwrap_or(T::zero());

    let drift = self.effective_drift();

    for i in 1..self.n {
      s[i] = s[i - 1]
        + (drift - self.lambda * self.k) * s[i - 1] * dt
        + s[i - 1] * v[i - 1].sqrt() * cgn1[i - 1]
        + s[i - 1] * jump_increments[i];

      let dv = (self.alpha - self.beta * v[i - 1]) * dt
        + self.sigma * v[i - 1].powf(T::from_f64_fast(0.5)) * cgn2[i - 1];

      v[i] = match self.use_sym.unwrap_or(false) {
        true => (v[i - 1] + dv).abs(),
        false => (v[i - 1] + dv).max(T::zero()),
      }
    }

    [s, v]
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyBates {
  inner_f32: Option<Bates1996<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<Bates1996<f64, crate::traits::CallableDist<f64>>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyBates {
  #[new]
  #[pyo3(signature = (lambda_, k, alpha, beta, sigma, rho, distribution, n, mu=None, b=None, r=None, r_f=None, s0=None, v0=None, t=None, use_sym=None, dtype=None))]
  fn new(
    lambda_: f64,
    k: f64,
    alpha: f64,
    beta: f64,
    sigma: f64,
    rho: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
    n: usize,
    mu: Option<f64>,
    b: Option<f64>,
    r: Option<f64>,
    r_f: Option<f64>,
    s0: Option<f64>,
    v0: Option<f64>,
    t: Option<f64>,
    use_sym: Option<bool>,
    dtype: Option<&str>,
  ) -> Self {
    use crate::stochastic::process::poisson::Poisson;
    match dtype.unwrap_or("f64") {
      "f32" => {
        let cpoisson = CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_ as f32, Some(n), t.map(|v| v as f32)),
        );
        Self {
          inner_f32: Some(Bates1996::new(
            mu.map(|v| v as f32),
            b.map(|v| v as f32),
            r.map(|v| v as f32),
            r_f.map(|v| v as f32),
            lambda_ as f32,
            k as f32,
            alpha as f32,
            beta as f32,
            sigma as f32,
            rho as f32,
            n,
            s0.map(|v| v as f32),
            v0.map(|v| v as f32),
            t.map(|v| v as f32),
            use_sym,
            cpoisson,
          )),
          inner_f64: None,
        }
      }
      _ => {
        let cpoisson = CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_, Some(n), t),
        );
        Self {
          inner_f32: None,
          inner_f64: Some(Bates1996::new(
            mu, b, r, r_f, lambda_, k, alpha, beta, sigma, rho, n, s0, v0, t, use_sym, cpoisson,
          )),
        }
      }
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    if let Some(ref inner) = self.inner_f64 {
      let [s, v] = inner.sample();
      (
        s.into_pyarray(py).into_py_any(py).unwrap(),
        v.into_pyarray(py).into_py_any(py).unwrap(),
      )
    } else if let Some(ref inner) = self.inner_f32 {
      let [s, v] = inner.sample();
      (
        s.into_pyarray(py).into_py_any(py).unwrap(),
        v.into_pyarray(py).into_py_any(py).unwrap(),
      )
    } else {
      unreachable!()
    }
  }

  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::ndarray::Array2;
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    if let Some(ref inner) = self.inner_f64 {
      let samples = inner.sample_par(m);
      let n = samples[0][0].len();
      let mut r0 = Array2::<f64>::zeros((m, n));
      let mut r1 = Array2::<f64>::zeros((m, n));
      for (i, [a, b]) in samples.iter().enumerate() {
        r0.row_mut(i).assign(a);
        r1.row_mut(i).assign(b);
      }
      (
        r0.into_pyarray(py).into_py_any(py).unwrap(),
        r1.into_pyarray(py).into_py_any(py).unwrap(),
      )
    } else if let Some(ref inner) = self.inner_f32 {
      let samples = inner.sample_par(m);
      let n = samples[0][0].len();
      let mut r0 = Array2::<f32>::zeros((m, n));
      let mut r1 = Array2::<f32>::zeros((m, n));
      for (i, [a, b]) in samples.iter().enumerate() {
        r0.row_mut(i).assign(a);
        r1.row_mut(i).assign(b);
      }
      (
        r0.into_pyarray(py).into_py_any(py).unwrap(),
        r1.into_pyarray(py).into_py_any(py).unwrap(),
      )
    } else {
      unreachable!()
    }
  }
}

#[cfg(test)]
mod tests {
  use rand_distr::Normal;

  use super::*;
  use crate::stochastic::process::poisson::Poisson;

  fn make_bates(
    mu: Option<f64>,
    b: Option<f64>,
    r: Option<f64>,
    r_f: Option<f64>,
  ) -> Bates1996<f64, Normal<f64>> {
    let cpoisson = CompoundPoisson::new(
      Normal::new(0.0, 1.0).expect("valid normal"),
      Poisson::new(1.0, Some(8), Some(1.0)),
    );
    Bates1996::new(
      mu,
      b,
      r,
      r_f,
      1.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      8,
      Some(1.0),
      Some(0.0),
      Some(1.0),
      Some(false),
      cpoisson,
    )
  }

  #[test]
  fn effective_drift_prefers_r_minus_rf_when_present() {
    let p = make_bates(Some(0.9), Some(0.7), Some(0.4), Some(0.1));
    assert!((p.effective_drift() - 0.3).abs() < 1e-12);
  }

  #[test]
  fn effective_drift_uses_b_if_rates_missing() {
    let p = make_bates(Some(0.9), Some(0.7), None, None);
    assert!((p.effective_drift() - 0.7).abs() < 1e-12);
  }

  #[test]
  fn effective_drift_falls_back_to_mu() {
    let p = make_bates(Some(0.9), None, None, None);
    assert!((p.effective_drift() - 0.9).abs() < 1e-12);
  }
}
