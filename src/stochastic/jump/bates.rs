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

    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);

    s[0] = self.s0.unwrap_or(T::zero());
    v[0] = self.v0.unwrap_or(T::zero());

    let drift = match (self.mu, self.b, self.r, self.r_f) {
      (Some(r), Some(r_f), ..) => r - r_f,
      (Some(b), ..) => b,
      _ => self.mu.unwrap(),
    };

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      s[i] = s[i - 1]
        + (drift - self.lambda * self.k) * s[i - 1] * dt
        + s[i - 1] * v[i - 1].sqrt() * cgn1[i - 1]
        + jumps.sum();

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
  inner: Bates1996<f64, crate::traits::CallableDist>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyBates {
  #[new]
  #[pyo3(signature = (lambda_, k, alpha, beta, sigma, rho, distribution, n, mu=None, b=None, r=None, r_f=None, s0=None, v0=None, t=None, use_sym=None))]
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
  ) -> Self {
    use crate::stochastic::process::poisson::Poisson;
    let cpoisson = CompoundPoisson::new(
      crate::traits::CallableDist::new(distribution),
      Poisson::new(lambda_, Some(n), t),
    );
    Self {
      inner: Bates1996::new(
        mu, b, r, r_f, lambda_, k, alpha, beta, sigma, rho, n, s0, v0, t, use_sym, cpoisson,
      ),
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    let [s, v] = self.inner.sample();
    (
      s.into_pyarray(py).into_py_any(py).unwrap(),
      v.into_pyarray(py).into_py_any(py).unwrap(),
    )
  }
}
