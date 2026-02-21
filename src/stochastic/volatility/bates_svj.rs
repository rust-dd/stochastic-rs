//! # Bates SVJ
//!
//! $$
//! \begin{aligned}
//! d\ln S_t &= (\mu - \lambda\kappa_J - \tfrac12 v_t)\,dt + \sqrt{v_t}\,dW_t + Z\,dN_t \\
//! dv_t &= (\alpha - \beta\,v_t)\,dt + \sigma\sqrt{v_t}\,dW_t^v
//! \end{aligned}
//! $$
//!
//! where $\langle dW, dW^v\rangle = \rho\,dt$, $N_t$ is Poisson with intensity $\lambda$,
//! and $Z\sim\mathcal{N}(\nu,\omega^2)$.
//!
use ndarray::Array1;
use rand_distr::Distribution;

use crate::distributions::normal::SimdNormal;
use crate::distributions::poisson::SimdPoisson;
use crate::stochastic::noise::cgns::CGNS;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct BatesSVJ<T: FloatExt> {
  /// Drift rate of the asset price
  pub mu: Option<T>,
  /// Cost-of-carry rate
  pub b: Option<T>,
  /// Domestic risk-free interest rate
  pub r: Option<T>,
  /// Foreign risk-free interest rate
  pub r_f: Option<T>,
  /// Jump intensity (Poisson arrival rate)
  pub lambda: T,
  /// Mean of the jump log-size Z ~ N(nu, omega^2)
  pub nu: T,
  /// Standard deviation of the jump log-size Z
  pub omega: T,
  /// Variance drift level (often kappa * theta in Heston notation)
  pub alpha: T,
  /// Variance mean-reversion speed (often kappa)
  pub beta: T,
  /// Volatility of variance (vol-of-vol)
  pub sigma: T,
  /// Correlation between asset and variance Brownian motions
  pub rho: T,
  /// Number of discrete time steps in the simulation grid
  pub n: usize,
  /// Initial asset price (must be > 0 for log-price simulation)
  pub s0: Option<T>,
  /// Initial variance level
  pub v0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted)
  pub t: Option<T>,
  /// Use symmetric (abs) instead of truncation (max(0)) for variance positivity
  pub use_sym: Option<bool>,
  cgns: CGNS<T>,
}

impl<T: FloatExt> BatesSVJ<T> {
  pub fn new(
    mu: Option<T>,
    b: Option<T>,
    r: Option<T>,
    r_f: Option<T>,
    lambda: T,
    nu: T,
    omega: T,
    alpha: T,
    beta: T,
    sigma: T,
    rho: T,
    n: usize,
    s0: Option<T>,
    v0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(
      rho >= -T::one() && rho <= T::one(),
      "rho must be in [-1, 1]"
    );
    assert!(omega >= T::zero(), "omega must be >= 0");
    assert!(lambda >= T::zero(), "lambda must be >= 0");

    Self {
      mu,
      b,
      r,
      r_f,
      lambda,
      nu,
      omega,
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
    }
  }

  #[inline]
  fn kappa_j(&self) -> T {
    (self.nu + T::from_f64_fast(0.5) * self.omega * self.omega).exp() - T::one()
  }

  #[inline]
  fn drift(&self) -> T {
    match (self.mu, self.b, self.r, self.r_f) {
      (_, _, Some(r), Some(r_f)) => r - r_f,
      (_, Some(b), _, _) => b,
      (Some(mu), _, _, _) => mu,
      _ => panic!("one of (r and r_f), b, or mu must be provided"),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for BatesSVJ<T> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = &self.cgns.sample();

    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);

    let s0 = self.s0.unwrap_or(T::one());
    assert!(s0 > T::zero(), "s0 must be > 0 for log-price simulation");
    s[0] = s0;

    v[0] = self.v0.unwrap_or(T::zero()).max(T::zero());

    let drift = self.drift();
    let kappa_j = self.kappa_j();

    let mut rng = rand::rng();

    let pois = if self.lambda > T::zero() {
      Some(SimdPoisson::<u32>::new(
        (self.lambda * dt).to_f64().unwrap(),
      ))
    } else {
      None
    };

    let z_std = SimdNormal::<f64, 64>::new(0.0, 1.0);

    for i in 1..self.n {
      let v_prev = match self.use_sym.unwrap_or(false) {
        true => v[i - 1].abs(),
        false => v[i - 1].max(T::zero()),
      };
      let sqrt_v = v_prev.sqrt();

      let mut jump_sum_z = T::zero();
      if let Some(pois) = &pois {
        let k: u32 = pois.sample(&mut rng);
        if k > 0 {
          let kf = T::from_usize_(k as usize);
          let z0: f64 = z_std.sample(&mut rng);
          jump_sum_z = self.nu * kf + self.omega * kf.sqrt() * T::from_f64_fast(z0);
        }
      }

      let log_inc = (drift - self.lambda * kappa_j - T::from_f64_fast(0.5) * v_prev) * dt
        + sqrt_v * cgn1[i - 1]
        + jump_sum_z;
      s[i] = s[i - 1] * log_inc.exp();

      let dv = (self.alpha - self.beta * v_prev) * dt + self.sigma * sqrt_v * cgn2[i - 1];
      v[i] = match self.use_sym.unwrap_or(false) {
        true => (v_prev + dv).abs(),
        false => (v_prev + dv).max(T::zero()),
      };
    }

    [s, v]
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn variance_stays_non_negative() {
    let p = BatesSVJ::new(
      Some(0.05_f64),
      None,
      None,
      None,
      0.5,
      -0.1,
      0.2,
      0.04,
      1.5,
      0.3,
      -0.7,
      256,
      Some(100.0),
      Some(0.04),
      Some(1.0),
      Some(false),
    );
    let [_s, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
  }

  #[test]
  fn price_stays_positive() {
    let p = BatesSVJ::new(
      Some(0.05_f64),
      None,
      None,
      None,
      0.5,
      -0.1,
      0.2,
      0.04,
      1.5,
      0.3,
      -0.7,
      256,
      Some(100.0),
      Some(0.04),
      Some(1.0),
      Some(false),
    );
    let [s, _v] = p.sample();
    assert!(s.iter().all(|x| *x > 0.0));
  }

  #[test]
  fn drift_prefers_r_minus_rf() {
    let p = BatesSVJ::new(
      Some(0.9_f64),
      Some(0.7),
      Some(0.4),
      Some(0.1),
      0.5,
      0.0,
      0.1,
      0.04,
      1.5,
      0.3,
      -0.5,
      8,
      Some(1.0),
      Some(0.04),
      Some(1.0),
      None,
    );
    assert!((p.drift() - 0.3).abs() < 1e-12);
  }

  #[test]
  fn drift_uses_b_if_rates_missing() {
    let p = BatesSVJ::new(
      Some(0.9_f64),
      Some(0.7),
      None,
      None,
      0.5,
      0.0,
      0.1,
      0.04,
      1.5,
      0.3,
      -0.5,
      8,
      Some(1.0),
      Some(0.04),
      Some(1.0),
      None,
    );
    assert!((p.drift() - 0.7).abs() < 1e-12);
  }

  #[test]
  fn drift_falls_back_to_mu() {
    let p = BatesSVJ::new(
      Some(0.9_f64),
      None,
      None,
      None,
      0.5,
      0.0,
      0.1,
      0.04,
      1.5,
      0.3,
      -0.5,
      8,
      Some(1.0),
      Some(0.04),
      Some(1.0),
      None,
    );
    assert!((p.drift() - 0.9).abs() < 1e-12);
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyBatesSVJ {
  inner_f32: Option<BatesSVJ<f32>>,
  inner_f64: Option<BatesSVJ<f64>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyBatesSVJ {
  #[new]
  #[pyo3(signature = (lambda_, nu, omega, alpha, beta, sigma, rho, n, mu=None, b=None, r=None, r_f=None, s0=None, v0=None, t=None, use_sym=None, dtype=None))]
  fn new(
    lambda_: f64,
    nu: f64,
    omega: f64,
    alpha: f64,
    beta: f64,
    sigma: f64,
    rho: f64,
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
    match dtype.unwrap_or("f64") {
      "f32" => Self {
        inner_f32: Some(BatesSVJ::new(
          mu.map(|v| v as f32),
          b.map(|v| v as f32),
          r.map(|v| v as f32),
          r_f.map(|v| v as f32),
          lambda_ as f32,
          nu as f32,
          omega as f32,
          alpha as f32,
          beta as f32,
          sigma as f32,
          rho as f32,
          n,
          s0.map(|v| v as f32),
          v0.map(|v| v as f32),
          t.map(|v| v as f32),
          use_sym,
        )),
        inner_f64: None,
      },
      _ => Self {
        inner_f32: None,
        inner_f64: Some(BatesSVJ::new(
          mu, b, r, r_f, lambda_, nu, omega, alpha, beta, sigma, rho, n, s0, v0, t, use_sym,
        )),
      },
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
    use numpy::IntoPyArray;
    use numpy::ndarray::Array2;
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
