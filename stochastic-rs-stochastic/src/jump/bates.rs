//! # Bates
//!
//! $$
//! \begin{aligned}dS_t&=(r-r_f-\lambda k)S_tdt+\sqrt{v_t}S_t dW_t^S+(Y-1)S_{t^-}dN_t\\dv_t&=\kappa(\theta-v_t)dt+\sigma\sqrt{v_t}dW_t^v\end{aligned}
//! $$
//!
use ndarray::Array1;
use rand_distr::Distribution;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::noise::cgns::Cgns;
use crate::process::cpoisson::CompoundPoisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[inline]
fn validate_drift_args<T: FloatExt>(
  mu: Option<T>,
  b: Option<T>,
  r: Option<T>,
  r_f: Option<T>,
  type_name: &'static str,
) {
  let has_r_pair = r.is_some() && r_f.is_some();
  if !(has_r_pair || b.is_some() || mu.is_some()) {
    panic!("{type_name}: one of (r and r_f), b, or mu must be provided");
  }
}

pub struct Bates1996<T, D, S: SeedExt = Unseeded>
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
  cgns: Cgns<T>,
  pub cpoisson: CompoundPoisson<T, D>,
  pub seed: S,
}

impl<T, D, S: SeedExt> Bates1996<T, D, S>
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
    seed: S,
  ) -> Self {
    if let Some(v0) = v0 {
      assert!(v0 >= T::zero(), "v0 must be non-negative");
    }
    validate_drift_args(mu, b, r, r_f, "Bates1996");

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
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
      cpoisson,
      seed,
    }
  }
}

impl<T, D, S: SeedExt> Bates1996<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  #[inline]
  fn effective_drift(&self) -> T {
    match (self.r, self.r_f, self.b, self.mu) {
      (Some(r), Some(r_f), _, _) => r - r_f,
      (_, _, Some(b), _) => b,
      (_, _, _, Some(mu)) => mu,
      _ => unreachable!("validate_drift_args ensures at least one of (r+r_f), b, mu is set"),
    }
  }
}

impl<T, D, S: SeedExt> ProcessExt<T> for Bates1996<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = &self.cgns.sample();
    let jump_increments = self.cpoisson.sample_grid_relative_increments(self.n, dt);

    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);

    s[0] = self.s0.unwrap_or(T::zero());
    v[0] = self.v0.unwrap_or(T::zero()).max(T::zero());

    let drift = self.effective_drift();

    for i in 1..self.n {
      let v_prev = v[i - 1].max(T::zero());
      s[i] = s[i - 1]
        + (drift - self.lambda * self.k) * s[i - 1] * dt
        + s[i - 1] * v_prev.sqrt() * cgn1[i - 1]
        + s[i - 1] * jump_increments[i];

      let dv = (self.alpha - self.beta * v_prev) * dt + self.sigma * v_prev.sqrt() * cgn2[i - 1];

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
  seeded_f32:
    Option<Bates1996<f32, crate::traits::CallableDist<f32>, crate::simd_rng::Deterministic>>,
  seeded_f64:
    Option<Bates1996<f64, crate::traits::CallableDist<f64>, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyBates {
  #[new]
  #[pyo3(signature = (lambda_, k, alpha, beta, sigma, rho, distribution, n, mu=None, b=None, r=None, r_f=None, s0=None, v0=None, t=None, use_sym=None, seed=None, dtype=None))]
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
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> Self {
    use crate::process::poisson::Poisson;
    let mut s = Self {
      inner_f32: None,
      inner_f64: None,
      seeded_f32: None,
      seeded_f64: None,
    };
    match dtype.unwrap_or("f64") {
      "f32" => {
        let cpoisson = CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_ as f32, Some(n), t.map(|v| v as f32), Unseeded),
          Unseeded,
        );
        match seed {
          Some(sd) => {
            s.seeded_f32 = Some(Bates1996::new(
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
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f32 = Some(Bates1996::new(
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
              Unseeded,
            ));
          }
        }
      }
      _ => {
        let cpoisson = CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_, Some(n), t, Unseeded),
          Unseeded,
        );
        match seed {
          Some(sd) => {
            s.seeded_f64 = Some(Bates1996::new(
              mu,
              b,
              r,
              r_f,
              lambda_,
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
              cpoisson,
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f64 = Some(Bates1996::new(
              mu, b, r, r_f, lambda_, k, alpha, beta, sigma, rho, n, s0, v0, t, use_sym, cpoisson,
              Unseeded,
            ));
          }
        }
      }
    }
    s
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
      let [a, b] = inner.sample();
      (
        a.into_pyarray(py).into_py_any(py).unwrap(),
        b.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }

  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use numpy::ndarray::Array2;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
      let samples = inner.sample_par(m);
      let n = samples[0][0].len();
      let mut r0 = Array2::zeros((m, n));
      let mut r1 = Array2::zeros((m, n));
      for (i, [a, b]) in samples.iter().enumerate() {
        r0.row_mut(i).assign(a);
        r1.row_mut(i).assign(b);
      }
      (
        r0.into_pyarray(py).into_py_any(py).unwrap(),
        r1.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }
}

#[cfg(test)]
mod tests {
  use rand_distr::Normal;

  use super::*;
  use crate::process::poisson::Poisson;

  fn make_bates(
    mu: Option<f64>,
    b: Option<f64>,
    r: Option<f64>,
    r_f: Option<f64>,
  ) -> Bates1996<f64, Normal<f64>> {
    let cpoisson = CompoundPoisson::new(
      Normal::new(0.0, 1.0).expect("valid normal"),
      Poisson::new(1.0, Some(8), Some(1.0), Unseeded),
      Unseeded,
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
      Unseeded,
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
