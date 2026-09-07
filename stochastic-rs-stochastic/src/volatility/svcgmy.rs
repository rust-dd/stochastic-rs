//! # Svcgmy (CGMYSV discrete-time approximation)
//!
//! Paper model — Kim Y. S. (2021), *Sample Path Generation of the
//! Stochastic Volatility CGMY Process and Its Application to
//! Path-Dependent Option Pricing*, Journal of Risk and Financial
//! Management 14(2), 77, DOI: 10.3390/jrfm14020077:
//! $$
//! L_t = Z_{V_t} + \rho v_t, \quad V_t = \int_0^t v_s ds,
//! $$
//! where $Z$ is a standard Cgmy process (independent of $v$) and $v$ follows Cir:
//! $$
//! dv_t=\kappa(\eta-v_t)dt+\zeta\sqrt{v_t}dW_t.
//! $$
//!
//! This implementation generates the discrete-time approximation on a grid
//! $t_m = m\Delta t$, using Algorithm 1 in the paper.
//!
//! Notes:
//! - `rho` is a **loading** on $v_t$ (not a correlation), so it is not restricted to [-1, 1].
//! - Series indices follow Algorithm 1: **j = 1..J**, with **Γ0 = 0**.
//!

use ndarray::Array1;
use scilib::math::basic::gamma;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::non_central_chi_squared::SimdNonCentralChiSquared;
use stochastic_rs_distributions::uniform::SimdUniform;

use crate::device::Cpu;
use crate::process::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Cgmy Stochastic Volatility process (CGMYSV)
///
/// Reference: Kim Y. S. (2021), DOI: 10.3390/jrfm14020077 (see the
/// module docs for the full citation).
#[derive(Clone)]
pub struct Svcgmy<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Positive tempering parameter λ+ > 0
  pub lambda_plus: T,
  /// Negative tempering parameter λ− > 0
  pub lambda_minus: T,
  /// Activity parameter α (0 < α < 2)
  pub alpha: T,
  /// Cir mean reversion κ > 0
  pub kappa: T,
  /// Cir long-term level η >= 0
  pub eta: T,
  /// Cir vol-of-vol ζ > 0
  pub zeta: T,
  /// Loading parameter ρ
  pub rho: T,
  /// Number of time steps (M+1 points including t=0)
  pub n: usize,
  /// Truncation level J (number of series terms)
  pub j: usize,
  /// Initial value (interpreted as L0)
  pub x0: Option<T>,
  /// Initial variance v0
  pub v0: Option<T>,
  /// Time horizon T
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Svcgmy<T, S> {
  pub fn new(
    lambda_plus: T,
    lambda_minus: T,
    alpha: T,
    kappa: T,
    eta: T,
    zeta: T,
    rho: T,
    n: usize,
    j: usize,
    x0: Option<T>,
    v0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(lambda_plus > T::zero(), "lambda_plus must be positive");
    assert!(lambda_minus > T::zero(), "lambda_minus must be positive");
    assert!(
      alpha > T::zero() && alpha < T::from_usize_(2),
      "alpha must be in (0, 2)"
    );
    assert!(kappa > T::zero(), "kappa must be positive");
    assert!(eta >= T::zero(), "eta must be non-negative");
    assert!(zeta > T::zero(), "zeta must be positive");
    assert!(n >= 2, "n must be >= 2");

    if let Some(v0) = v0 {
      assert!(v0 >= T::zero(), "v0 must be non-negative");
    }

    Self {
      backend: Cpu,
      lambda_plus,
      lambda_minus,
      alpha,
      kappa,
      eta,
      zeta,
      rho,
      n,
      j,
      x0,
      v0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Svcgmy<T, S, B> {
  /// The horizon, one when omitted.
  fn t_max(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The grid spacing.
  fn dt(&self) -> T {
    self.t_max() / T::from_usize_(self.n - 1)
  }

  /// The CIR step's degrees of freedom, `4 κ η / ζ²`.
  fn degrees_of_freedom(&self) -> T {
    T::from_usize_(4) * self.kappa * self.eta / self.zeta.powi(2)
  }

  /// The tempered-stable scale `C = (Γ(2 − α)(λ₊^{α−2} + λ₋^{α−2}))^{-1}`.
  fn tempering_constant(&self) -> T {
    let two = T::from_usize_(2);
    T::one()
      / (T::from_f64_fast(gamma(2.0 - self.alpha.to_f64().unwrap()))
        * (self.lambda_plus.powf(self.alpha - two) + self.lambda_minus.powf(self.alpha - two)))
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for Svcgmy<T, S, B>
{
  /// The host's constants folded once: the arrival bound's rate at unit
  /// variance, the drift per unit variance, and the CIR step's `2c` and decay.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let two = T::from_usize_(2);
    let ek = (-self.kappa * self.dt()).exp();
    let c = two * self.kappa / ((T::one() - ek) * self.zeta.powi(2));
    let bcoef = -(self.lambda_plus.powf(self.alpha - T::one())
      - self.lambda_minus.powf(self.alpha - T::one()))
      / ((T::one() - self.alpha)
        * (self.lambda_plus.powf(self.alpha - two) + self.lambda_minus.powf(self.alpha - two)));
    crate::euler::EulerSpec::StochasticVolatilityCgmy {
      rate0: self.alpha / (two * self.tempering_constant() * self.t_max()),
      inv_alpha: T::one() / self.alpha,
      lambda_plus: self.lambda_plus,
      lambda_minus: self.lambda_minus,
      bcoef,
      twoc: two * c,
      ek,
      rho: self.rho,
    }
  }

  /// The jump part starts at `x0 − ρ v0`, so the reported log-price starts
  /// at `x0`; the variance at `v0`.
  fn initial_state(&self) -> [T; 4] {
    let v0 = self.v0.unwrap_or(T::zero());
    [
      self.x0.unwrap_or(T::zero()) - self.rho * v0,
      v0,
      T::zero(),
      T::zero(),
    ]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t_max()
  }

  fn time_step(&self) -> T {
    self.dt()
  }

  /// The central χ²(df − 1) of the exact CIR step, a gamma of shape
  /// `(df − 1) / 2` and scale two; none at exactly one degree of freedom,
  /// where the step is the shifted normal's square alone.
  fn gamma_draws(&self) -> Option<crate::euler::GammaDraws<T>> {
    let df = self.degrees_of_freedom();
    (df > T::one()).then(|| crate::euler::GammaDraws {
      first: (
        (df - T::one()) / T::from_usize_(2),
        T::from_usize_(2),
        T::zero(),
      ),
      second: None,
    })
  }

  /// One term per series index, as many as the host draws.
  fn series_terms(&self) -> Option<u32> {
    Some(self.j as u32)
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Svcgmy<T, S> { lambda_plus, lambda_minus, alpha, kappa, eta, zeta, rho, n, j, x0, v0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Svcgmy<T, S, B> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = SvcgmySampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> SvcgmySampler<T, S> {
    SvcgmySampler {
      lambda_plus: self.lambda_plus,
      lambda_minus: self.lambda_minus,
      alpha: self.alpha,
      kappa: self.kappa,
      eta: self.eta,
      zeta: self.zeta,
      rho: self.rho,
      n: self.n,
      j: self.j,
      x0: self.x0.unwrap_or(T::zero()),
      v0: self.v0.unwrap_or(T::zero()),
      t: self.t,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine when the variance's exact step has a degree of
  /// freedom to spare and the series fits the kernels' slots; anything else
  /// keeps the process on the host, chunked exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    if self.device_ready() {
      self.backend.system_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self.backend.system_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    if self.device_ready() {
      self.backend.system_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_system_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_system_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: the variance's exact step needs
  /// at least one degree of freedom — below it the host draws a Poisson
  /// mixture the kernels do not carry — and the series terms fit the kernels'
  /// per-path slots. Anything else samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.degrees_of_freedom() >= T::one() && self.j <= crate::euler::SERIES_SLOTS))
      .then_some("degrees of freedom below one, or a series past the kernels' cells")
  }
}

/// Reusable [`Svcgmy`] sampling state: owns the scalar parameters and the seed
/// source. The Cir noncentral-χ² driver, the series uniforms/exponentials and
/// the arrival-time Poisson generator are rebuilt per fill in the legacy
/// seed-consumption order, so the first call reproduces the original stream
/// bit-for-bit.
#[doc(hidden)]
pub struct SvcgmySampler<T: FloatExt, S: SeedExt> {
  lambda_plus: T,
  lambda_minus: T,
  alpha: T,
  kappa: T,
  eta: T,
  zeta: T,
  rho: T,
  n: usize,
  j: usize,
  x0: T,
  v0: T,
  t: Option<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> SvcgmySampler<T, S> {
  #[allow(non_snake_case)]
  fn fill_paths(&mut self, x: &mut [T], v: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(self.n - 1);

    let mut y = Array1::<T>::zeros(self.n);

    x[0] = self.x0;
    v[0] = self.v0;
    // y = L - rho * v  =>  L = y + rho * v
    y[0] = x[0] - self.rho * v[0];

    let f2 = T::from_usize_(2);

    // C = (Γ(2-α) (λ+^(α-2) + λ-^(α-2)))^{-1}
    let g = gamma(2.0 - self.alpha.to_f64().unwrap());
    let C = T::one()
      / (T::from_f64_fast(g)
        * (self.lambda_plus.powf(self.alpha - f2) + self.lambda_minus.powf(self.alpha - f2)));

    // Cir exact-step constants (paper)
    let exp_kdt = (-self.kappa * dt).exp();
    let c = (f2 * self.kappa) / ((T::one() - exp_kdt) * self.zeta.powi(2));
    let df = T::from_usize_(4) * self.kappa * self.eta / self.zeta.powi(2);

    // 1) Simulate v on the grid via noncentral chi-square
    let nchi2 = SimdNonCentralChiSquared::<T>::new(df, &self.seed);
    for i in 1..self.n {
      let ncp = f2 * c * v[i - 1] * exp_kdt;
      v[i] = nchi2.sample_ncp(ncp) / (f2 * c);
    }

    // 2) Series random variables (Algorithm 1 uses j=1..J with Γ0=0)
    let J = self.j;
    let size = J + 1; // index 0 is reserved (Γ0=0)

    let uniform = SimdUniform::<T>::new(T::zero(), T::one(), &self.seed);
    let exp = SimdExp::<T>::new(T::one(), &self.seed);

    // U_j ~ Unif(0,1), E_j ~ Exp(1), τ_j ~ Unif(0,T)
    let mut U = Array1::<T>::zeros(size);
    uniform.fill_slice(U.as_slice_mut().unwrap());
    let E = Array1::from_shape_fn(size, |_| exp.sample_fast());
    let mut tau_raw = Array1::<T>::zeros(size);
    uniform.fill_slice(tau_raw.as_slice_mut().unwrap());
    let tau = tau_raw * t_max;

    // Γ_0=0, Γ_j = Γ_{j-1} + E'_j; we reuse Poisson-generator-as-arrival-times for Γ_j.
    // Derives (not `Unseeded`) so Γ_j is reproducible under a `Deterministic` seed and
    // distinct path-to-path: `self.seed` is this sampler's own already chunk-decorrelated
    // basis (see `sampler()`), so ticking it once more per fill stays safely confined to
    // this chunk's own sequence.
    let P = Poisson::new(T::one(), Some(size), None, self.seed.derive()).sample();

    // c(τ_j) = C * v_{k-1} where (k-1)dt < τ_j <= k dt
    let mut c_tau = Array1::<T>::zeros(size);
    for j in 1..size {
      let tau_j = tau[j];
      let k = ((tau_j / dt).ceil()).min(T::from_usize_(self.n - 1));
      let v_km1 = if k == T::zero() {
        v[0]
      } else {
        v[k.to_usize().unwrap() - 1]
      };
      c_tau[j] = C * v_km1;
    }

    // 3) Build Y on the grid (Algorithm 1)
    for i in 1..self.n {
      // b_m = - v_{m-1} (λ+^(α-1) - λ-^(α-1)) / ((1-α)(λ+^(α-2)+λ-^(α-2)))
      let numerator = v[i - 1]
        * (self.lambda_plus.powf(self.alpha - T::one())
          - self.lambda_minus.powf(self.alpha - T::one()));
      let denominator = (T::one() - self.alpha)
        * (self.lambda_plus.powf(self.alpha - f2) + self.lambda_minus.powf(self.alpha - f2));
      let b = -numerator / denominator;

      let mut jump_component = T::zero();

      let t_1 = T::from_usize_(i - 1) * dt;
      let t = T::from_usize_(i) * dt;

      for j in 1..=J {
        if tau[j] > t_1 && tau[j] <= t {
          // V_j is chosen as λ+ or -λ- with prob 1/2
          let v_j = if uniform.sample_fast() < T::from_f64_fast(0.5) {
            self.lambda_plus
          } else {
            -self.lambda_minus
          };

          // min term: ((α Γ_j)/(2 c(τ_j) T))^{-1/α}  ∧  E_j U_j^{1/α} / |V_j|
          let num = self.alpha * P[j];
          let den = f2 * c_tau[j] * t_max;
          let term1 = (num / den).powf(-T::one() / self.alpha);

          let term2 = E[j] * U[j].powf(T::one() / self.alpha) / v_j.abs();
          let min_term = term1.min(term2);

          jump_component += min_term * (v_j / v_j.abs());
        }
      }

      y[i] = y[i - 1] + jump_component + b * dt;
    }

    // 4) L ≈ Y + ρ v  (paper Eq. (7))
    for i in 1..self.n {
      x[i] = y[i] + self.rho * v[i];
    }
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for SvcgmySampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [x, v] = out;
    self.fill_paths(
      x.as_slice_mut().expect("Svcgmy output must be contiguous"),
      v.as_slice_mut().expect("Svcgmy output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut x = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    self.fill_paths(
      x.as_slice_mut().expect("contiguous"),
      v.as_slice_mut().expect("contiguous"),
    );
    [x, v]
  }
}

py_process_2x1d!(PySvcgmy, Svcgmy,
  sig: (lambda_plus, lambda_minus, alpha, kappa, eta, zeta, rho, n, j, x0=None, v0=None, t=None, seed=None, dtype=None),
  params: (lambda_plus: f64, lambda_minus: f64, alpha: f64, kappa: f64, eta: f64, zeta: f64, rho: f64, n: usize, j: usize, x0: Option<f64>, v0: Option<f64>, t: Option<f64>),
  device
);
