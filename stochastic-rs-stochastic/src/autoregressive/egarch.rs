//! # Egarch
//!
//! $$
//! \log(\sigma_t^2)=\omega+\sum_{i=1}^p[\alpha_i(|z_{t-i}|-\mathbb E|z|)+\gamma_i z_{t-i}]
//! +\sum_{j=1}^q\beta_j\log(\sigma_{t-j}^2)
//! $$
//!
//! Reference: Nelson (1991), Econometrica 59(2), 347–370, DOI 10.2307/2938260

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

/// EGARCH(p, q) on standardized residuals $z_t = X_t/\sigma_t$ with $E|z| = \sqrt{2/\pi}$;
/// `alpha` loads the magnitude, `gamma` the sign, and `new` enforces a stationary `beta`.
#[derive(Debug, Clone)]
pub struct Egarch<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Constant term (\(\omega\)) in log-variance
  pub omega: T,
  /// Magnitude effect coefficients (\(\alpha_1, \ldots, \alpha_p\))
  pub alpha: Array1<T>,
  /// Sign/asymmetry effect coefficients (\(\gamma_1, \ldots, \gamma_p\))
  pub gamma: Array1<T>,
  /// Persistence coefficients for log-variance (\(\beta_1, \ldots, \beta_q\))
  pub beta: Array1<T>,
  /// Number of observations
  pub n: usize,
  /// The seed: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Egarch<T, S> {
  /// Nelson's (1991) EGARCH(p, q). Panics unless `1 − Σ βⱼ xʲ` has its roots outside the unit
  /// circle (`|beta| < 1` at order one); a log-variance beyond `exp`'s range samples `∞` or NaN.
  pub fn new(
    omega: T,
    alpha: Array1<T>,
    gamma: Array1<T>,
    beta: Array1<T>,
    n: usize,
    seed: S,
  ) -> Self {
    assert!(
      omega.is_finite(),
      "omega must satisfy `omega.is_finite()`, got omega = {omega:?}"
    );
    assert_entries("alpha", "alpha[i].is_finite()", &alpha, |v| v.is_finite());
    assert_entries("gamma", "gamma[i].is_finite()", &gamma, |v| v.is_finite());
    assert_entries("beta", "beta[i].is_finite()", &beta, |v| v.is_finite());
    assert!(n >= 1, "n must satisfy `n >= 1`, got n = {n:?}");
    assert!(
      alpha.len() == gamma.len(),
      "alpha.len() must satisfy `alpha.len() == gamma.len()`, got alpha.len() = {:?}, gamma.len() = {:?}",
      alpha.len(),
      gamma.len()
    );
    check_beta(&beta);
    Self {
      backend: Cpu,
      omega,
      alpha,
      gamma,
      beta,
      n,
      seed,
    }
  }
}

/// Panics unless `1 − Σ βⱼ xʲ` has every root outside the unit circle: the step-down recursion's
/// partial autocorrelations all lie in `(−1, 1)` (Barndorff-Nielsen & Schou 1973).
#[track_caller]
fn check_beta<T: FloatExt>(beta: &Array1<T>) {
  let mut a = beta.to_vec();
  let stationary = (1..=a.len()).rev().all(|k| {
    let r = a[k - 1];
    let lower = (0..k - 1)
      .map(|j| (a[j] + r * a[k - 2 - j]) / (T::one() - r * r))
      .collect::<Vec<_>>();
    a[..k - 1].copy_from_slice(&lower);
    r.abs() < T::one()
  });
  assert!(
    stationary,
    "beta must satisfy `1 - Σ beta[j] x^(j+1) has every root outside the unit circle`, got beta = {:?}",
    beta.to_vec()
  );
}

/// The Euler engine's view of EGARCH(1,1). Higher orders read back more than
/// one lag, which the engine's four state slots do not bound.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Egarch<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    assert!(
      self.alpha.len() == 1 && self.gamma.len() == 1 && self.beta.len() == 1,
      "Egarch: a launch carries order (1, 1); sample through `ProcessExt`, which keeps a higher order on the host"
    );
    crate::euler::EulerSpec::ExponentialGarch {
      omega: self.omega,
      alpha: self.alpha[0],
      gamma: self.gamma[0],
      beta: self.beta[0],
      e_abs_z: T::from_f64_fast((2.0_f64 / std::f64::consts::PI).sqrt()),
    }
  }

  fn initial_value(&self) -> T {
    T::zero()
  }

  /// The series starts at `σ₀ z₀` with `ln σ₀² = ω`, so the launch steps
  /// before writing its first point.
  fn initial_state(&self) -> [T; 4] {
    check_beta(&self.beta);
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

  /// A discrete-time model advances by one observation.
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

backend_switch!([T: FloatExt, S: SeedExt] Egarch<T, S> { omega, alpha, gamma, beta, n, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Egarch<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Egarch<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = EgarchSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> EgarchSampler<T> {
    EgarchSampler {
      n: self.n,
      omega: self.omega,
      alpha: self.alpha.clone(),
      gamma: self.gamma.clone(),
      beta: self.beta.clone(),
      normal: SimdNormal::<T>::new(T::zero(), T::one()).seeded(&self.seed),
    }
  }

  /// What a device runs: order (1, 1), since the family carries one lag of each; a higher order
  /// stays on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.alpha.len() == 1 && self.gamma.len() == 1 && self.beta.len() == 1))
      .then_some("an order above (1, 1): the family carries one lag of each")
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

/// Reusable [`Egarch`] sampling state: it owns the innovation source and the coefficients, so a
/// Monte-Carlo loop pays the `SimdNormal` setup once.
#[doc(hidden)]
#[derive(Debug)]
pub struct EgarchSampler<T: FloatExt> {
  n: usize,
  omega: T,
  alpha: Array1<T>,
  gamma: Array1<T>,
  beta: Array1<T>,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> EgarchSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let n = out.len();
    let p = self.alpha.len();
    let q = self.beta.len();

    // Generate white noise z_t ~ N(0,1)
    let mut z = Array1::<T>::zeros(n);
    if n > 0 {
      let slice = z.as_slice_mut().expect("contiguous");
      self.normal.fill_slice(slice);
    }

    // Scratch array for log of variance (the output buffer holds X_t)
    let mut log_sigma2 = Array1::<T>::zeros(n);

    // For normal(0,1), the expected absolute value is sqrt(2/pi)
    let e_abs_z = T::from_f64_fast((2.0_f64 / std::f64::consts::PI).sqrt());
    check_beta(&self.beta);

    for t in 0..n {
      if t == 0 {
        // Initialize log-variance (e.g., with omega)
        log_sigma2[t] = self.omega;
      } else {
        // 1) Compute the shock term from p lags
        let mut shock_term = T::zero();
        for i in 1..=p {
          if t >= i {
            // Standardized residual from step t-i
            let sigma_t_i = (log_sigma2[t - i].exp()).sqrt();
            let z_t_i = out[t - i] / sigma_t_i; // z_{t-i}

            // Add alpha_i(|z_{t-i}| - E|z|) + gamma_i z_{t-i}
            shock_term += self.alpha[i - 1] * (z_t_i.abs() - e_abs_z) + self.gamma[i - 1] * z_t_i;
          }
        }

        // 2) Sum in the log-variance from q lags
        let mut persistence_term = T::zero();
        for j in 1..=q {
          if t >= j {
            persistence_term += self.beta[j - 1] * log_sigma2[t - j];
          }
        }

        // 3) Final log-variance
        log_sigma2[t] = self.omega + shock_term + persistence_term;
      }

      // Convert log_sigma2[t] to sigma_t and compute X_t
      let sigma_t = (log_sigma2[t].exp()).sqrt();
      out[t] = sigma_t * z[t];
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for EgarchSampler<T> {}

impl<T: FloatExt> PathSampler<T> for EgarchSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Egarch output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyEgarch, Egarch,
  sig: (omega, alpha, gamma_, beta, n, seed=None, dtype=None),
  params: (omega: f64, alpha: Vec<f64>, gamma_: Vec<f64>, beta: Vec<f64>, n: usize),
  device
);
