//! # Ar
//!
//! $$
//! X_t=\sum_{k=1}^p \phi_k X_{t-k}+\varepsilon_t,\qquad \varepsilon_t\sim\mathcal N(0,\sigma^2)
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Implements an AR(p) model:
///
/// \[
///   X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \dots + \phi_p X_{t-p}
///         + \epsilon_t,
///   \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2).
/// \]
///
/// # Fields
/// - `phi`: Vector of AR coefficients (\(\phi_1, \ldots, \phi_p\)).
/// - `sigma`: Standard deviation of the noise \(\epsilon_t\).
/// - `n`: Length of the time series.
/// - `m`: Optional batch size (for parallel sampling).
/// - `x0`: Optional array of initial values. If provided, should have length at least `phi.len()`.
#[derive(Debug, Clone)]
pub struct ARp<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// AR coefficients
  pub phi: Array1<T>,
  /// Noise std dev
  pub sigma: T,
  /// Number of observations
  pub n: usize,
  /// Optional initial conditions
  pub x0: Option<Array1<T>>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> ARp<T, S> {
  /// Create a new AR process with given coefficients and noise standard deviation.
  pub fn new(phi: Array1<T>, sigma: T, n: usize, x0: Option<Array1<T>>, seed: S) -> Self {
    assert!(sigma > T::zero(), "ARp requires sigma > 0");
    if let Some(init) = &x0 {
      let required = phi.len().min(n);
      assert!(
        init.len() >= required,
        "ARp requires x0.len() >= min(phi.len(), n) (got {}, need at least {})",
        init.len(),
        required
      );
    }
    Self {
      backend: Cpu,
      phi,
      sigma,
      n,
      x0,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> ARp<T, S, B> {}

/// The Euler engine's view of an autoregression. Higher orders read back more
/// than one lag, which the engine's four state slots do not bound.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for ARp<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    assert!(
      self.phi.len() == 1,
      "ARp reaches a device at order 1 only; this one is order {}",
      self.phi.len()
    );
    crate::euler::EulerSpec::Autoregressive {
      phi: self.phi[0],
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    T::zero()
  }

  /// A process given an initial value starts from it; one without starts from
  /// its first draw, which is why the flag below is conditional.
  fn initial_state(&self) -> [T; 4] {
    let x0 = self.x0.as_ref().and_then(|v| v.first().copied());
    [x0.unwrap_or(T::zero()), T::zero(), T::zero(), T::zero()]
  }

  /// Every grid point is a draw, so the launch steps before writing the
  /// first.
  fn step_first(&self) -> bool {
    self.x0.is_none()
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    T::from_usize_(self.n)
  }

  fn time_step(&self) -> T {
    T::one()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] ARp<T, S> { phi, sigma, n, x0, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for ARp<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = ARpSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> ARpSampler<T> {
    ARpSampler {
      n: self.n,
      phi: self.phi.clone(),
      x0: self.x0.clone(),
      normal: SimdNormal::<T>::new(T::zero(), self.sigma, &self.seed),
    }
  }

  /// Through the Euler engine: on a device the draw happens in the kernel, on
  /// the host devices it is this process's own sampler, chunked exactly as
  /// `ProcessExt` chunks.
  fn sample(&self) -> Array1<T> {
    self.backend.euler_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self.backend.euler_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    self.backend.euler_paths(self, m)
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    self.backend.try_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    self.backend.try_euler_paths(self, m)
  }
}

/// Reusable [`ARp`] sampling state: owns the Gaussian innovation source and the
/// model coefficients so a Monte-Carlo loop pays the `SimdNormal` setup once.
#[doc(hidden)]
pub struct ARpSampler<T: FloatExt> {
  n: usize,
  phi: Array1<T>,
  x0: Option<Array1<T>>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> ARpSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let n = out.len();
    let p = self.phi.len();

    let mut noise = Array1::<T>::zeros(n);
    if n > 0 {
      let slice = noise.as_slice_mut().expect("contiguous");
      self.normal.fill_slice(slice);
    }
    let noise = &noise;

    // Fill initial conditions if provided
    if let Some(init) = &self.x0 {
      // Copy up to min(p, n)
      for i in 0..p.min(n) {
        out[i] = init[i];
      }
    }

    // AR recursion
    let start = if self.x0.is_some() { p.min(n) } else { 0 };
    for t in start..n {
      let mut val = T::zero();
      for k in 1..=p {
        if t >= k {
          val += self.phi[k - 1] * out[t - k];
        }
      }
      out[t] = val + noise[t];
    }
  }
}

impl<T: FloatExt> PathSampler<T> for ARpSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("ARp output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyARp, ARp,
  sig: (phi, sigma, n, x0=None, seed=None, dtype=None),
  params: (phi: Vec<f64>, sigma: f64, n: usize, x0: Option<Vec<f64>>)
);
