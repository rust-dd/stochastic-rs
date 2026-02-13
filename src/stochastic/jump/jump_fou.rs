use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::process::cpoisson::CompoundPoisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct JumpFOU<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub hurst: T,
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub cpoisson: CompoundPoisson<T, D>,
  fgn: FGN<T>,
}

impl<T, D> JumpFOU<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    cpoisson: CompoundPoisson<T, D>,
  ) -> Self {
    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      cpoisson,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T, D> ProcessExt<T> for JumpFOU<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.fgn.dt();
    let fgn = &self.fgn.sample();

    let mut jump_fou = Array1::<T>::zeros(self.n);
    jump_fou[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      jump_fou[i] = jump_fou[i - 1]
        + self.theta * (self.mu - jump_fou[i - 1]) * dt
        + self.sigma * fgn[i - 1]
        + jumps.sum();
    }

    jump_fou
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyJumpFOU {
  inner: JumpFOU<f64, crate::traits::CallableDist>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyJumpFOU {
  #[new]
  #[pyo3(signature = (hurst, theta, mu, sigma, distribution, lambda_, n, x0=None, t=None))]
  fn new(
    hurst: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
    lambda_: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
  ) -> Self {
    use crate::stochastic::process::cpoisson::CompoundPoisson;
    use crate::stochastic::process::poisson::Poisson;
    let cpoisson = CompoundPoisson::new(
      crate::traits::CallableDist::new(distribution),
      Poisson::new(lambda_, Some(n), t),
    );
    Self {
      inner: JumpFOU::new(hurst, theta, mu, sigma, n, x0, t, cpoisson),
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    self
      .inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap()
  }
}
