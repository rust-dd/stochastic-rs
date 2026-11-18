//! The variance-level threshold recursion of Glosten, Jagannathan and Runkle, not Zakoian's
//! TGARCH, which thresholds the standard deviation $\sigma_t$ and is not equivalent to it.
//!
//! $$
//! \sigma_t^2=\omega+\sum_{i=1}^p(\alpha_i+\gamma_i\mathbf 1_{\{X_{t-i}<0\}})X_{t-i}^2
//! +\sum_{j=1}^q\beta_j\sigma_{t-j}^2
//! $$
//!
//! Reference: Glosten et al. (1993), JF 48(5), 1779–1801, DOI 10.1111/j.1540-6261.1993.tb05128.x
//! Reference: Zakoian (1994), JEDC 18(5), 931–955, DOI 10.1016/0165-1889(94)90039-6

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::autoregressive::threshold::variance_denominator;
use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::precondition::assert_entries;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// GJR-GARCH(p, q): `gamma` adds to `alpha` on a fall, and the path starts at its unconditional
/// variance; `new` enforces the stationarity `Σα + Σγ/2 + Σβ < 1`.
#[derive(Debug, Clone)]
pub struct GjrGarch<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Constant term in conditional variance dynamics.
  pub omega: T,
  /// Arch coefficients α_i (positive-squared-residual loading), length p.
  pub alpha: Array1<T>,
  /// Threshold extra loading γ_i applied when `X_{t-i}<0`, length p
  /// (must match `alpha.len()`).
  pub gamma: Array1<T>,
  /// Garch coefficients β_j (past-variance persistence loading), length q.
  pub beta: Array1<T>,
  /// Length of the generated time series.
  pub n: usize,
  /// The seed: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> GjrGarch<T, S> {
  /// The GJR (1993) threshold GARCH(p, q). Panics unless `omega > 0`, every `alpha[i] >= 0`,
  /// `alpha[i] + gamma[i] >= 0` and `beta[j] >= 0`, and `Σα + Σγ/2 + Σβ < 1`.
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
      omega > T::zero(),
      "omega must satisfy `omega > T::zero()`, got omega = {omega:?}"
    );
    assert!(
      alpha.len() == gamma.len(),
      "alpha.len() must satisfy `alpha.len() == gamma.len()`, got alpha.len() = {:?}, gamma.len() = {:?}",
      alpha.len(),
      gamma.len()
    );
    variance_denominator(&alpha, &gamma, &beta, "gamma");
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

/// The Euler engine's view of GJR-GARCH(1,1).
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for GjrGarch<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    assert!(
      self.alpha.len() == 1 && self.gamma.len() == 1 && self.beta.len() == 1,
      "GjrGarch: a launch carries order (1, 1); sample through `ProcessExt`, which keeps a higher order on the host"
    );
    crate::euler::EulerSpec::ThresholdGarch {
      omega: self.omega,
      alpha: self.alpha[0],
      gamma: self.gamma[0],
      beta: self.beta[0],
    }
  }

  fn initial_value(&self) -> T {
    T::zero()
  }

  /// The series starts at `σ₀ z₀`, so the launch steps before it writes its first point: the
  /// initial state is the seed variance and the not-yet-warm marker, not a reported value.
  fn initial_state(&self) -> [T; 4] {
    let denom = variance_denominator(&self.alpha, &self.gamma, &self.beta, "gamma");
    [T::zero(), self.omega / denom, T::zero(), T::zero()]
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

backend_switch!([T: FloatExt, S: SeedExt] GjrGarch<T, S> { omega, alpha, gamma, beta, n, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for GjrGarch<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for GjrGarch<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = GjrGarchSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> GjrGarchSampler<T> {
    GjrGarchSampler {
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

/// Reusable [`GjrGarch`] sampling state: it owns the innovation source and the coefficients, so
/// a Monte-Carlo loop pays the `SimdNormal` setup once.
#[doc(hidden)]
#[derive(Debug)]
pub struct GjrGarchSampler<T: FloatExt> {
  n: usize,
  omega: T,
  alpha: Array1<T>,
  gamma: Array1<T>,
  beta: Array1<T>,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> GjrGarchSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let n = out.len();
    let p = self.alpha.len();
    let q = self.beta.len();

    // Standard normal noise
    let mut z = Array1::<T>::zeros(n);
    if n > 0 {
      let slice = z.as_slice_mut().expect("contiguous");
      self.normal.fill_slice(slice);
    }

    // Scratch array for sigma_t^2 (the output buffer holds X_t)
    let mut sigma2 = Array1::<T>::zeros(n);
    let var_floor = T::from_f64_fast(1e-12);

    let denom = variance_denominator(&self.alpha, &self.gamma, &self.beta, "gamma");

    for t in 0..n {
      if t == 0 {
        sigma2[t] = self.omega / denom;
      } else {
        let mut var_t = self.omega;

        // Sum over p lags
        for i in 1..=p {
          if t >= i {
            let x_lag = out[t - i];
            // Threshold indicator
            let indicator = if x_lag < T::zero() {
              T::one()
            } else {
              T::zero()
            };

            // alpha_i * X_{t-i}^2 + gamma_i * X_{t-i}^2 * indicator
            var_t +=
              self.alpha[i - 1] * x_lag.powi(2) + self.gamma[i - 1] * x_lag.powi(2) * indicator;
          }
        }

        // Sum over q lags
        for j in 1..=q {
          if t >= j {
            var_t += self.beta[j - 1] * sigma2[t - j];
          }
        }

        sigma2[t] = var_t;
      }
      // X_t = sigma_t * z_t
      out[t] = sigma2[t].max(var_floor).sqrt() * z[t];
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for GjrGarchSampler<T> {}

impl<T: FloatExt> PathSampler<T> for GjrGarchSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("GjrGarch output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyGjrGarch, GjrGarch,
  sig: (omega, alpha, gamma_, beta, n, seed=None, dtype=None),
  params: (omega: f64, alpha: Vec<f64>, gamma_: Vec<f64>, beta: Vec<f64>, n: usize),
  device
);
