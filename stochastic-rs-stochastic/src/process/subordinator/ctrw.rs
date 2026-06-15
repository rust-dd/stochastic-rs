use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::alpha_stable::SimdAlphaStable;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::uniform::SimdUniform;

use super::sample_positive_stable;
use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Waiting-time distribution for Ctrw.
pub enum CtrwWaitingLaw<T: FloatExt> {
  Exponential { rate: T },
  Gamma { shape: T, rate: T },
  InverseGaussian { mu: T, lambda: T },
  PositiveStable { alpha: T, scale: T },
}

/// Jump-size distribution for Ctrw.
pub enum CtrwJumpLaw<T: FloatExt> {
  Normal { mean: T, std: T },
  SymmetricStable { alpha: T, scale: T },
  Rademacher { scale: T },
}

/// Continuous-time random walk sampled on a fixed output grid.
pub struct Ctrw<T: FloatExt, S: SeedExt = Unseeded> {
  pub waiting: CtrwWaitingLaw<T>,
  pub jumps: CtrwJumpLaw<T>,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Ctrw<T, S> {
  pub fn new(
    waiting: CtrwWaitingLaw<T>,
    jumps: CtrwJumpLaw<T>,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      waiting,
      jumps,
      n,
      x0,
      t,
      seed,
    }
  }
}

enum WaitingSampler<T: FloatExt> {
  Exp(SimdExp<T>),
  Gamma(SimdGamma<T>),
  Ig(SimdInverseGauss<T>),
  PosStable { alpha: f64, scale: f64 },
}

enum JumpSampler<T: FloatExt> {
  Normal(SimdNormal<T>),
  Stable(SimdAlphaStable<T>),
  Rademacher(T),
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Ctrw<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = CtrwSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> CtrwSampler<T> {
    let x0 = self.x0.unwrap_or(T::zero());
    let n_increments = self.n.saturating_sub(1).max(1);
    let t_max = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let dt = t_max / n_increments as f64;

    let waiting = match self.waiting {
      CtrwWaitingLaw::Exponential { rate } => {
        assert!(
          rate > T::zero(),
          "Ctrw Exponential waiting requires rate > 0"
        );
        WaitingSampler::Exp(SimdExp::new(rate, &self.seed))
      }
      CtrwWaitingLaw::Gamma { shape, rate } => {
        assert!(
          shape > T::zero() && rate > T::zero(),
          "Ctrw Gamma waiting requires shape > 0 and rate > 0"
        );
        WaitingSampler::Gamma(SimdGamma::<T>::new(shape, T::one() / rate, &self.seed))
      }
      CtrwWaitingLaw::InverseGaussian { mu, lambda } => {
        assert!(
          mu > T::zero() && lambda > T::zero(),
          "Ctrw Ig waiting requires mu > 0 and lambda > 0"
        );
        WaitingSampler::Ig(SimdInverseGauss::<T>::new(mu, lambda, &self.seed))
      }
      CtrwWaitingLaw::PositiveStable { alpha, scale } => {
        assert!(
          alpha > T::zero() && alpha < T::one() && scale > T::zero(),
          "Ctrw positive-stable waiting requires alpha in (0,1) and scale > 0"
        );
        WaitingSampler::PosStable {
          alpha: alpha.to_f64().unwrap(),
          scale: scale.to_f64().unwrap(),
        }
      }
    };

    let jumps = match self.jumps {
      CtrwJumpLaw::Normal { mean, std } => {
        assert!(std > T::zero(), "Ctrw normal jumps require std > 0");
        JumpSampler::Normal(SimdNormal::new(mean, std, &self.seed))
      }
      CtrwJumpLaw::SymmetricStable { alpha, scale } => {
        assert!(
          alpha > T::zero() && alpha <= T::from_usize_(2) && scale > T::zero(),
          "Ctrw stable jumps require alpha in (0,2] and scale > 0"
        );
        JumpSampler::Stable(SimdAlphaStable::<T>::new(
          alpha,
          T::zero(),
          scale,
          T::zero(),
          &self.seed,
        ))
      }
      CtrwJumpLaw::Rademacher { scale } => {
        assert!(scale > T::zero(), "Ctrw rademacher jumps require scale > 0");
        JumpSampler::Rademacher(scale)
      }
    };

    let uniform = SimdUniform::<f64>::new(0.0, 1.0, &self.seed);

    CtrwSampler {
      n: self.n,
      x0,
      dt,
      waiting,
      jumps,
      uniform,
    }
  }
}

/// Reusable [`Ctrw`] sampling state: the owned waiting-time and jump-size
/// distribution drivers plus the shared uniform source (used for positive-
/// stable waiting times and Rademacher jumps).
#[doc(hidden)]
pub struct CtrwSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: f64,
  waiting: WaitingSampler<T>,
  jumps: JumpSampler<T>,
  uniform: SimdUniform<f64>,
}

impl<T: FloatExt> CtrwSampler<T> {
  fn draw_wait(&self) -> f64 {
    match &self.waiting {
      WaitingSampler::Exp(d) => d.sample_fast().to_f64().unwrap(),
      WaitingSampler::Gamma(d) => d.sample_fast().to_f64().unwrap(),
      WaitingSampler::Ig(d) => d.sample_fast().to_f64().unwrap(),
      WaitingSampler::PosStable { alpha, scale } => {
        scale * sample_positive_stable(*alpha, &self.uniform)
      }
    }
    .max(1e-12)
  }

  fn draw_jump(&self) -> f64 {
    match &self.jumps {
      JumpSampler::Normal(d) => d.sample_fast().to_f64().unwrap(),
      JumpSampler::Stable(d) => d.sample_fast().to_f64().unwrap(),
      JumpSampler::Rademacher(scale) => {
        if self.uniform.sample_fast() < 0.5 {
          scale.to_f64().unwrap()
        } else {
          -scale.to_f64().unwrap()
        }
      }
    }
  }

  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }

    let mut x = self.x0.to_f64().unwrap();
    let mut next_event = self.draw_wait();

    for i in 1..out.len() {
      let t_i = i as f64 * self.dt;
      let mut safety = 0usize;
      while next_event <= t_i {
        x += self.draw_jump();
        next_event += self.draw_wait();
        safety += 1;
        if safety > 1_000_000 {
          break;
        }
      }
      out[i] = T::from_f64_fast(x);
    }
  }
}

impl<T: FloatExt> PathSampler<T> for CtrwSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Ctrw output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyCtrw {
  inner_f32: Option<Ctrw<f32>>,
  inner_f64: Option<Ctrw<f64>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyCtrw {
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
        "PyCtrw: invalid waiting_law '{}' — expected one of 'exponential' | 'gamma' | 'inverse_gaussian' | 'inversegaussian' | 'ig' | 'stable' | 'positive_stable' | 'positivestable'",
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
        "PyCtrw: invalid jump_law '{}' — expected one of 'normal' | 'symmetric_stable' | 'symmetricstable' | 'stable' | 'rademacher'",
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
          inner_f32: Some(Ctrw::new(
            waiting_f32,
            jumps_f32,
            n,
            x0.map(|v| v as f32),
            t.map(|v| v as f32),
            Unseeded,
          )),
          inner_f64: None,
        }
      }
      _ => Self {
        inner_f32: None,
        inner_f64: Some(Ctrw::new(waiting_f64, jumps_f64, n, x0, t, Unseeded)),
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
