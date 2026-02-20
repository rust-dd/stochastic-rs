use ndarray::Array1;
use rand::Rng;
use rand_distr::Distribution;

use super::sample_positive_stable;
use crate::distributions::alpha_stable::SimdAlphaStable;
use crate::distributions::exp::SimdExp;
use crate::distributions::gamma::SimdGamma;
use crate::distributions::inverse_gauss::SimdInverseGauss;
use crate::distributions::normal::SimdNormal;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Waiting-time distribution for CTRW.
pub enum CtrwWaitingLaw<T: FloatExt> {
  Exponential { rate: T },
  Gamma { shape: T, rate: T },
  InverseGaussian { mu: T, lambda: T },
  PositiveStable { alpha: T, scale: T },
}

/// Jump-size distribution for CTRW.
pub enum CtrwJumpLaw<T: FloatExt> {
  Normal { mean: T, std: T },
  SymmetricStable { alpha: T, scale: T },
  Rademacher { scale: T },
}

/// Continuous-time random walk sampled on a fixed output grid.
pub struct CTRW<T: FloatExt> {
  pub waiting: CtrwWaitingLaw<T>,
  pub jumps: CtrwJumpLaw<T>,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
}

impl<T: FloatExt> CTRW<T> {
  pub fn new(
    waiting: CtrwWaitingLaw<T>,
    jumps: CtrwJumpLaw<T>,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
  ) -> Self {
    Self {
      waiting,
      jumps,
      n,
      x0,
      t,
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for CTRW<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut out = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return out;
    }
    let x0 = self.x0.unwrap_or(T::zero());
    out[0] = x0;
    if self.n == 1 {
      return out;
    }

    enum WaitingSampler<T: FloatExt> {
      Exp(SimdExp<T>),
      Gamma(SimdGamma<T>),
      IG(SimdInverseGauss<T>),
      PosStable { alpha: f64, scale: f64 },
    }
    enum JumpSampler<T: FloatExt> {
      Normal(SimdNormal<T>),
      Stable(SimdAlphaStable<T>),
      Rademacher(T),
    }

    let waiting_sampler = match self.waiting {
      CtrwWaitingLaw::Exponential { rate } => {
        assert!(
          rate > T::zero(),
          "CTRW Exponential waiting requires rate > 0"
        );
        WaitingSampler::Exp(SimdExp::new(rate))
      }
      CtrwWaitingLaw::Gamma { shape, rate } => {
        assert!(
          shape > T::zero() && rate > T::zero(),
          "CTRW Gamma waiting requires shape > 0 and rate > 0"
        );
        WaitingSampler::Gamma(SimdGamma::new(shape, T::one() / rate))
      }
      CtrwWaitingLaw::InverseGaussian { mu, lambda } => {
        assert!(
          mu > T::zero() && lambda > T::zero(),
          "CTRW IG waiting requires mu > 0 and lambda > 0"
        );
        WaitingSampler::IG(SimdInverseGauss::new(mu, lambda))
      }
      CtrwWaitingLaw::PositiveStable { alpha, scale } => {
        assert!(
          alpha > T::zero() && alpha < T::one() && scale > T::zero(),
          "CTRW positive-stable waiting requires alpha in (0,1) and scale > 0"
        );
        WaitingSampler::PosStable {
          alpha: alpha.to_f64().unwrap(),
          scale: scale.to_f64().unwrap(),
        }
      }
    };

    let jump_sampler = match self.jumps {
      CtrwJumpLaw::Normal { mean, std } => {
        assert!(std > T::zero(), "CTRW normal jumps require std > 0");
        JumpSampler::Normal(SimdNormal::new(mean, std))
      }
      CtrwJumpLaw::SymmetricStable { alpha, scale } => {
        assert!(
          alpha > T::zero() && alpha <= T::from_usize_(2) && scale > T::zero(),
          "CTRW stable jumps require alpha in (0,2] and scale > 0"
        );
        JumpSampler::Stable(SimdAlphaStable::new(alpha, T::zero(), scale, T::zero()))
      }
      CtrwJumpLaw::Rademacher { scale } => {
        assert!(scale > T::zero(), "CTRW rademacher jumps require scale > 0");
        JumpSampler::Rademacher(scale)
      }
    };

    let t_max = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let dt = t_max / (self.n - 1) as f64;
    let mut rng = rand::rng();
    let mut x = x0.to_f64().unwrap();

    let mut next_event = match &waiting_sampler {
      WaitingSampler::Exp(d) => d.sample(&mut rng).to_f64().unwrap(),
      WaitingSampler::Gamma(d) => d.sample(&mut rng).to_f64().unwrap(),
      WaitingSampler::IG(d) => d.sample(&mut rng).to_f64().unwrap(),
      WaitingSampler::PosStable { alpha, scale } => {
        scale * sample_positive_stable(*alpha, &mut rng)
      }
    }
    .max(1e-12);

    for i in 1..self.n {
      let t_i = i as f64 * dt;
      let mut safety = 0usize;
      while next_event <= t_i {
        let jump = match &jump_sampler {
          JumpSampler::Normal(d) => d.sample(&mut rng).to_f64().unwrap(),
          JumpSampler::Stable(d) => d.sample(&mut rng).to_f64().unwrap(),
          JumpSampler::Rademacher(scale) => {
            if rng.random_bool(0.5) {
              scale.to_f64().unwrap()
            } else {
              -scale.to_f64().unwrap()
            }
          }
        };
        x += jump;

        let wait = match &waiting_sampler {
          WaitingSampler::Exp(d) => d.sample(&mut rng).to_f64().unwrap(),
          WaitingSampler::Gamma(d) => d.sample(&mut rng).to_f64().unwrap(),
          WaitingSampler::IG(d) => d.sample(&mut rng).to_f64().unwrap(),
          WaitingSampler::PosStable { alpha, scale } => {
            scale * sample_positive_stable(*alpha, &mut rng)
          }
        }
        .max(1e-12);
        next_event += wait;
        safety += 1;
        if safety > 1_000_000 {
          break;
        }
      }
      out[i] = T::from_f64_fast(x);
    }

    out
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyCTRW {
  inner_f32: Option<CTRW<f32>>,
  inner_f64: Option<CTRW<f64>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyCTRW {
  #[new]
  #[pyo3(signature = (
    waiting_law,
    waiting_p1,
    waiting_p2=None,
    jump_law="normal",
    jump_p1=0.0,
    jump_p2=None,
    n=1000,
    x0=None,
    t=None,
    dtype=None
  ))]
  fn new(
    waiting_law: &str,
    waiting_p1: f64,
    waiting_p2: Option<f64>,
    jump_law: &str,
    jump_p1: f64,
    jump_p2: Option<f64>,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    dtype: Option<&str>,
  ) -> Self {
    let waiting_f64 = match waiting_law.to_ascii_lowercase().as_str() {
      "exp" | "exponential" => CtrwWaitingLaw::Exponential { rate: waiting_p1 },
      "gamma" => CtrwWaitingLaw::Gamma {
        shape: waiting_p1,
        rate: waiting_p2.unwrap_or(1.0),
      },
      "ig" | "inverse_gaussian" | "inversegaussian" => CtrwWaitingLaw::InverseGaussian {
        mu: waiting_p1,
        lambda: waiting_p2.unwrap_or(1.0),
      },
      "stable" | "positive_stable" | "positivestable" => CtrwWaitingLaw::PositiveStable {
        alpha: waiting_p1,
        scale: waiting_p2.unwrap_or(1.0),
      },
      _ => panic!(
        "invalid waiting_law '{}'; expected one of: exponential, gamma, inverse_gaussian, positive_stable",
        waiting_law
      ),
    };

    let jumps_f64 = match jump_law.to_ascii_lowercase().as_str() {
      "normal" => CtrwJumpLaw::Normal {
        mean: jump_p1,
        std: jump_p2.unwrap_or(1.0),
      },
      "stable" | "symmetric_stable" | "symmetricstable" => CtrwJumpLaw::SymmetricStable {
        alpha: jump_p1,
        scale: jump_p2.unwrap_or(1.0),
      },
      "rademacher" => CtrwJumpLaw::Rademacher {
        scale: jump_p1.abs(),
      },
      _ => panic!(
        "invalid jump_law '{}'; expected one of: normal, symmetric_stable, rademacher",
        jump_law
      ),
    };

    match dtype.unwrap_or("f64") {
      "f32" => {
        let waiting_f32 = match waiting_f64 {
          CtrwWaitingLaw::Exponential { rate } => CtrwWaitingLaw::Exponential { rate: rate as f32 },
          CtrwWaitingLaw::Gamma { shape, rate } => CtrwWaitingLaw::Gamma {
            shape: shape as f32,
            rate: rate as f32,
          },
          CtrwWaitingLaw::InverseGaussian { mu, lambda } => CtrwWaitingLaw::InverseGaussian {
            mu: mu as f32,
            lambda: lambda as f32,
          },
          CtrwWaitingLaw::PositiveStable { alpha, scale } => CtrwWaitingLaw::PositiveStable {
            alpha: alpha as f32,
            scale: scale as f32,
          },
        };
        let jumps_f32 = match jumps_f64 {
          CtrwJumpLaw::Normal { mean, std } => CtrwJumpLaw::Normal {
            mean: mean as f32,
            std: std as f32,
          },
          CtrwJumpLaw::SymmetricStable { alpha, scale } => CtrwJumpLaw::SymmetricStable {
            alpha: alpha as f32,
            scale: scale as f32,
          },
          CtrwJumpLaw::Rademacher { scale } => CtrwJumpLaw::Rademacher {
            scale: scale as f32,
          },
        };
        Self {
          inner_f32: Some(CTRW::new(
            waiting_f32,
            jumps_f32,
            n,
            x0.map(|v| v as f32),
            t.map(|v| v as f32),
          )),
          inner_f64: None,
        }
      }
      _ => Self {
        inner_f32: None,
        inner_f64: Some(CTRW::new(waiting_f64, jumps_f64, n, x0, t)),
      },
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;

    if let Some(ref inner) = self.inner_f64 {
      inner.sample().into_pyarray(py).into_py_any(py).unwrap()
    } else if let Some(ref inner) = self.inner_f32 {
      inner.sample().into_pyarray(py).into_py_any(py).unwrap()
    } else {
      unreachable!()
    }
  }

  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use numpy::ndarray::Array2;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;

    if let Some(ref inner) = self.inner_f64 {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::<f64>::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py).into_py_any(py).unwrap()
    } else if let Some(ref inner) = self.inner_f32 {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::<f32>::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py).into_py_any(py).unwrap()
    } else {
      unreachable!()
    }
  }
}
