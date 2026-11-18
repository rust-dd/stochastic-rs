//! # Arch
//!
//! $$
//! \sigma_t^2=\omega+\sum_{i=1}^p\alpha_iX_{t-i}^2,\qquad X_t=\sigma_t z_t,\ z_t\sim\mathcal N(0,1)
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::precondition::assert_entries;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// ARCH(p): the conditional variance is `omega` plus the `alpha`-weighted last `p` squared
/// observations, started at `omega`.
#[derive(Debug, Clone)]
pub struct Arch<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Omega (constant term in variance)
  pub omega: T,
  /// Coefficients alpha_i
  pub alpha: Array1<T>,
  /// Length of series
  pub n: usize,
  /// Seed: [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Arch<T, S> {
  /// Engle's (1982) ARCH(p). Panics unless `n >= 1` and `omega > 0`, `alpha[i] >= 0` are finite;
  /// loadings that sum to one or more are accepted, and their variance may reach `∞` or NaN.
  pub fn new(omega: T, alpha: Array1<T>, n: usize, seed: S) -> Self {
    assert!(
      omega.is_finite(),
      "omega must satisfy `omega.is_finite()`, got omega = {omega:?}"
    );
    assert_entries("alpha", "alpha[i].is_finite()", &alpha, |v| v.is_finite());
    assert!(n >= 1, "n must satisfy `n >= 1`, got n = {n:?}");
    assert!(
      omega > T::zero(),
      "omega must satisfy `omega > T::zero()`, got omega = {omega:?}"
    );
    check_alpha(&alpha);
    Self {
      backend: Cpu,
      omega,
      alpha,
      n,
      seed,
    }
  }
}

/// One check for `new`, the samplers and a launch, which a field written after `new` reaches.
#[track_caller]
fn check_alpha<T: FloatExt>(alpha: &Array1<T>) {
  assert_entries("alpha", "alpha[i] >= T::zero()", alpha, |v| v >= T::zero());
}

/// The Euler engine's view of ARCH(1). Higher orders read back more than one
/// lag, which the engine's four state slots do not bound.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Arch<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    assert!(
      self.alpha.len() == 1,
      "Arch: a launch carries order one; sample through `ProcessExt`, which keeps a higher order on the host"
    );
    crate::euler::EulerSpec::Garch {
      omega: self.omega,
      alpha: self.alpha[0],
      beta: T::zero(),
    }
  }

  fn initial_value(&self) -> T {
    T::zero()
  }

  /// The series starts at `σ₀ z₀`, so the launch steps before it writes its first point: the
  /// initial state is the seed variance and the not-yet-warm marker, not a reported value.
  fn initial_state(&self) -> [T; 4] {
    check_alpha(&self.alpha);
    [T::zero(), self.omega, T::zero(), T::zero()]
  }

  fn step_first(&self) -> bool {
    true
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    T::from_usize_(self.n)
  }

  /// A discrete-time model advances by one observation, not by a fraction of
  /// a horizon.
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

backend_switch!([T: FloatExt, S: SeedExt] Arch<T, S> { omega, alpha, n, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Arch<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Arch<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = ArchSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> ArchSampler<T> {
    ArchSampler {
      n: self.n,
      omega: self.omega,
      alpha: self.alpha.clone(),
      normal: SimdNormal::<T>::new(T::zero(), T::one()).seeded(&self.seed),
    }
  }

  /// What a device runs: order one, since the family carries one lag; a higher order stays on
  /// the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.alpha.len() == 1)).then_some("an order above one: the family carries a single lag")
  }

  /// Through the Euler engine at the order the family carries; a higher order
  /// keeps the process on the host, chunked exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      self.backend.euler_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map_view(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, |path| f(path.view()))
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      self.backend.euler_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_euler_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }
}

/// Reusable [`Arch`] sampling state: it owns the innovation source and the coefficients, so a
/// Monte-Carlo loop pays the `SimdNormal` setup once.
#[doc(hidden)]
#[derive(Debug)]
pub struct ArchSampler<T: FloatExt> {
  n: usize,
  omega: T,
  alpha: Array1<T>,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> ArchSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let n = out.len();
    let m = self.alpha.len();

    let mut z = Array1::<T>::zeros(n);
    if n > 0 {
      let slice = z.as_slice_mut().expect("contiguous");
      self.normal.fill_slice(slice);
    }
    let var_floor = T::from_f64_fast(1e-12);
    check_alpha(&self.alpha);

    for t in 0..n {
      // compute sigma_t^2
      let mut var_t = self.omega;
      for i in 1..=m {
        if t >= i {
          let x_lag = out[t - i];
          var_t += self.alpha[i - 1] * x_lag.powi(2);
        }
      }
      let sigma_t = var_t.max(var_floor).sqrt();
      out[t] = sigma_t * z[t];
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for ArchSampler<T> {}

impl<T: FloatExt> PathSampler<T> for ArchSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Arch output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyArch, Arch,
  sig: (omega, alpha, n, seed=None, dtype=None),
  params: (omega: f64, alpha: Vec<f64>, n: usize),
  device
);
