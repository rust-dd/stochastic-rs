use ndarray::Array1;

use super::HestonPow;
use crate::stochastic::noise::cgns::CGNS;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Heston<T: FloatExt> {
  /// Initial stock price
  pub s0: Option<T>,
  /// Initial volatility
  pub v0: Option<T>,
  /// Mean reversion rate
  pub kappa: T,
  /// Long-run average volatility
  pub theta: T,
  /// Volatility of volatility
  pub sigma: T,
  /// Correlation between the stock price and its volatility
  pub rho: T,
  /// Drift of the stock price
  pub mu: T,
  /// Number of time steps
  pub n: usize,
  /// Time to maturity
  pub t: Option<T>,
  /// Power of the variance
  /// If 0.5 then it is the original Heston model
  /// If 1.5 then it is the 3/2 model
  pub pow: HestonPow,
  /// Use the symmetric method for the variance to avoid negative values
  pub use_sym: Option<bool>,
  /// Noise generator
  cgns: CGNS<T>,
}

impl<T: FloatExt> Heston<T> {
  pub fn new(
    s0: Option<T>,
    v0: Option<T>,
    kappa: T,
    theta: T,
    sigma: T,
    rho: T,
    mu: T,
    n: usize,
    t: Option<T>,
    pow: HestonPow,
    use_sym: Option<bool>,
  ) -> Self {
    assert!(kappa >= T::zero(), "kappa must be non-negative");
    assert!(theta >= T::zero(), "theta must be non-negative");
    assert!(sigma >= T::zero(), "sigma must be non-negative");
    if let Some(v0) = v0 {
      assert!(v0 >= T::zero(), "v0 must be non-negative");
    }

    Self {
      s0,
      v0,
      kappa,
      theta,
      sigma,
      rho,
      mu,
      n,
      t,
      pow,
      use_sym,
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for Heston<T> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = &self.cgns.sample();

    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);

    s[0] = self.s0.unwrap_or(T::zero());
    v[0] = self.v0.unwrap_or(T::zero()).max(T::zero());

    for i in 1..self.n {
      let v_prev = v[i - 1].max(T::zero());
      s[i] = s[i - 1] + self.mu * s[i - 1] * dt + s[i - 1] * v_prev.sqrt() * cgn1[i - 1];

      let dv = self.kappa * (self.theta - v_prev) * dt
        + self.sigma
          * v_prev.powf(match self.pow {
            HestonPow::Sqrt => T::from_f64_fast(0.5),
            HestonPow::ThreeHalves => T::from_f64_fast(1.5),
          })
          * cgn2[i - 1];

      v[i] = match self.use_sym.unwrap_or(false) {
        true => (v[i - 1] + dv).abs(),
        false => (v[i - 1] + dv).max(T::zero()),
      }
    }

    [s, v]
  }
}

impl<T: FloatExt> Heston<T> {
  /// Malliavin derivative of the volatility
  ///
  /// The Malliavin derivative of the Heston model is given by
  /// D_r v_t = \sigma v_t^{1/2} / 2 * exp(-(\kappa \theta / 2 - \sigma^2 / 8) / v_t * dt)
  ///
  /// The Malliavin derivative of the 3/2 Heston model is given by
  /// D_r v_t = \sigma v_t^{3/2} / 2 * exp(-(\kappa \theta / 2 + 3 \sigma^2 / 8) * v_t * dt)
  pub fn malliavin_of_vol(&self) -> [Array1<T>; 3] {
    let [s, v] = self.sample();
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);

    let mut det_term = Array1::zeros(self.n);
    let mut malliavin = Array1::zeros(self.n);
    let f2 = T::from_usize_(2);

    for i in 0..self.n {
      match self.pow {
        HestonPow::Sqrt => {
          det_term[i] = ((-(self.kappa * self.theta / f2
            - self.sigma.powi(2) / T::from_usize_(8))
            * (T::one() / *v.last().unwrap())
            - self.kappa / f2)
            * (T::from_usize_(self.n - i) * dt))
            .exp();
          malliavin[i] = (self.sigma * v.last().unwrap().sqrt() / f2) * det_term[i];
        }
        HestonPow::ThreeHalves => {
          det_term[i] = ((-(self.kappa * self.theta / f2
            + T::from_usize_(3) * self.sigma.powi(2) / T::from_usize_(8))
            * *v.last().unwrap()
            - (self.kappa * self.theta) / f2)
            * (T::from_usize_(self.n - i) * dt))
            .exp();
          malliavin[i] =
            (self.sigma * v.last().unwrap().powf(T::from_f64_fast(1.5)) / f2) * det_term[i];
        }
      };
    }

    [s, v, malliavin]
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  #[should_panic(expected = "v0 must be non-negative")]
  fn negative_initial_variance_panics() {
    let _ = Heston::new(
      Some(100.0_f64),
      Some(-0.1),
      1.0,
      0.04,
      0.3,
      -0.5,
      0.0,
      8,
      Some(1.0),
      HestonPow::Sqrt,
      Some(false),
    );
  }

  #[test]
  fn variance_path_stays_non_negative() {
    let p = Heston::new(
      Some(100.0_f64),
      Some(0.04),
      1.5,
      0.04,
      0.5,
      -0.7,
      0.0,
      128,
      Some(1.0),
      HestonPow::Sqrt,
      Some(false),
    );
    let [_s, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyHeston {
  inner_f32: Option<Heston<f32>>,
  inner_f64: Option<Heston<f64>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHeston {
  #[new]
  #[pyo3(signature = (kappa, theta, sigma, rho, mu, n, s0=None, v0=None, t=None, pow=None, use_sym=None, dtype=None))]
  fn new(
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    mu: f64,
    n: usize,
    s0: Option<f64>,
    v0: Option<f64>,
    t: Option<f64>,
    pow: Option<&str>,
    use_sym: Option<bool>,
    dtype: Option<&str>,
  ) -> Self {
    let hp = match pow.unwrap_or("sqrt") {
      "three_halves" | "3/2" => HestonPow::ThreeHalves,
      _ => HestonPow::Sqrt,
    };
    match dtype.unwrap_or("f64") {
      "f32" => Self {
        inner_f32: Some(Heston::new(
          s0.map(|v| v as f32),
          v0.map(|v| v as f32),
          kappa as f32,
          theta as f32,
          sigma as f32,
          rho as f32,
          mu as f32,
          n,
          t.map(|v| v as f32),
          hp,
          use_sym,
        )),
        inner_f64: None,
      },
      _ => Self {
        inner_f32: None,
        inner_f64: Some(Heston::new(
          s0, v0, kappa, theta, sigma, rho, mu, n, t, hp, use_sym,
        )),
      },
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    if let Some(ref inner) = self.inner_f64 {
      let [a, b] = inner.sample();
      (
        a.into_pyarray(py).into_py_any(py).unwrap(),
        b.into_pyarray(py).into_py_any(py).unwrap(),
      )
    } else if let Some(ref inner) = self.inner_f32 {
      let [a, b] = inner.sample();
      (
        a.into_pyarray(py).into_py_any(py).unwrap(),
        b.into_pyarray(py).into_py_any(py).unwrap(),
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
