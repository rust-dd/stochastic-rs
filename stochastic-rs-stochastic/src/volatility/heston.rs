//! # Heston
//!
//! $$
//! \begin{aligned}dS_t&=\mu S_tdt+\sqrt{v_t}S_tdW_t^S\\dv_t&=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,\ d\langle W^S,W^v\rangle_t=\rho dt\end{aligned}
//! $$
//!
use std::marker::PhantomData;

use ndarray::Array1;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::HestonPow;
use crate::device::Cpu;
use crate::device::HostBackend;
use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

mod scheme;
pub use scheme::AndersenQe;
pub use scheme::Euler;
pub use scheme::HestonScheme;

#[derive(Clone)]
pub struct Heston<T: FloatExt, S: SeedExt = Unseeded, Sch: HestonScheme = Euler, B = Cpu> {
  /// Initial stock price
  pub s0: Option<T>,
  /// Initial variance v₀ — a variance, not a volatility: `dv_t` above is
  /// the process this seeds, and `sqrt(v_t)` (not `v_t` itself) is the
  /// instantaneous volatility fed to `dS_t`.
  pub v0: Option<T>,
  /// Mean reversion rate
  pub kappa: T,
  /// Long-run variance level (θ in the module header's `dv_t` equation) —
  /// a variance, not a volatility, for the same reason as `v0`.
  pub theta: T,
  /// Volatility of volatility
  pub sigma: T,
  /// Correlation between the stock price and its volatility
  pub rho: T,
  /// Drift of the stock price
  pub mu: T,
  /// Number of time steps
  pub n: usize,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Power of the variance
  /// If 0.5 then it is the original Heston model
  /// If 1.5 then it is the 3/2 model
  pub pow: HestonPow,
  /// Use the symmetric method for the variance to avoid negative values
  pub use_sym: Option<bool>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// Noise generator (used by the [`Euler`] scheme; [`AndersenQe`] draws its
  /// own noise and leaves this untouched).
  cgns: Cgns<T>,
  /// Zero-sized marker for the compile-time variance scheme.
  _scheme: PhantomData<Sch>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Heston<T, S, Euler> {
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
    seed: S,
  ) -> Self {
    assert!(kappa >= T::zero(), "kappa must be non-negative");
    assert!(theta >= T::zero(), "theta must be non-negative");
    assert!(sigma >= T::zero(), "sigma must be non-negative");
    if let Some(v0) = v0 {
      assert!(v0 >= T::zero(), "v0 must be non-negative");
    }

    Self {
      backend: Cpu,
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
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
      _scheme: PhantomData,
    }
  }
}

/// s₀=100, v₀=0.04, κ=2.0, θ=0.04, σ=0.3, ρ=-0.7, μ=0.05 — a
/// parameterization of the Heston (1993) stochastic-volatility model
/// satisfying the Feller condition (`2κθ = 0.16 ≥ σ² = 0.09`). t=1, n=252
/// — one trading year of daily steps (this crate's `Default` convention).
impl<T: FloatExt> Default for Heston<T, Unseeded, Euler> {
  fn default() -> Self {
    Self::new(
      Some(T::from_f64_fast(100.0)),
      Some(T::from_f64_fast(0.04)),
      T::from_f64_fast(2.0),
      T::from_f64_fast(0.04),
      T::from_f64_fast(0.3),
      T::from_f64_fast(-0.7),
      T::from_f64_fast(0.05),
      252,
      Some(T::one()),
      HestonPow::Sqrt,
      Some(false),
      Unseeded,
    )
  }
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Heston::default().with_kappa(3.0).with_rho(-0.8)`. Implemented
/// generically over `Sch: HestonScheme` (not just [`Euler`], the only
/// scheme `new()`/`Default` produce) since `cgns` and every other field are
/// scheme-independent — this is what lets the setters chain after
/// [`qe()`](Heston::qe) too, e.g.
/// `Heston::default().qe().with_kappa(3.0)`.
impl<T: FloatExt, S: SeedExt, Sch: HestonScheme, B> Heston<T, S, Sch, B> {
  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: Option<T>) -> Self {
    self.s0 = s0;
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: Option<T>) -> Self {
    if let Some(v) = v0 {
      assert!(v >= T::zero(), "v0 must be non-negative");
    }
    self.v0 = v0;
    self
  }

  /// Replace `kappa`, all else unchanged.
  pub fn with_kappa(mut self, kappa: T) -> Self {
    assert!(kappa >= T::zero(), "kappa must be non-negative");
    self.kappa = kappa;
    self
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    assert!(theta >= T::zero(), "theta must be non-negative");
    self.theta = theta;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    assert!(sigma >= T::zero(), "sigma must be non-negative");
    self.sigma = sigma;
    self
  }

  /// Replace `rho`; rebuilds the cached correlated-Gaussian generator
  /// (`cgns`) so the new correlation actually reaches the sampler instead
  /// of a stale one computed from the old `rho`.
  pub fn with_rho(mut self, rho: T) -> Self {
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    self.mu = mu;
    self
  }

  /// Replace the number of simulation steps `n`; rebuilds the cached
  /// correlated-Gaussian generator, whose length and step size derive
  /// from `n`.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self.cgns = Cgns::new(self.rho, n - 1, self.t, Unseeded);
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the cached
  /// correlated-Gaussian generator's step size, which derives from `t`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self.cgns = Cgns::new(self.rho, self.n - 1, t, Unseeded);
    self
  }

  /// Replace `pow`, all else unchanged.
  pub fn with_pow(mut self, pow: HestonPow) -> Self {
    self.pow = pow;
    self
  }

  /// Replace `use_sym`, all else unchanged.
  pub fn with_use_sym(mut self, use_sym: Option<bool>) -> Self {
    self.use_sym = use_sym;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> Heston<T, S, Euler, B> {
  /// Switch to the [`AndersenQe`] variance scheme at compile time. Consumes
  /// the model and re-tags it — zero runtime cost (the fields are moved and
  /// the marker swapped). QE is defined for the square-root (CIR) variance,
  /// so keep `pow = HestonPow::Sqrt`.
  pub fn qe(self) -> Heston<T, S, AndersenQe> {
    Heston {
      backend: Cpu,
      s0: self.s0,
      v0: self.v0,
      kappa: self.kappa,
      theta: self.theta,
      sigma: self.sigma,
      rho: self.rho,
      mu: self.mu,
      n: self.n,
      t: self.t,
      pow: self.pow,
      use_sym: self.use_sym,
      seed: self.seed,
      cgns: self.cgns,
      _scheme: PhantomData,
    }
  }
}

/// The Euler engine's view of the Heston model. Only the [`Euler`] scheme has
/// a device form: the quadratic-exponential scheme draws from a
/// non-central chi-square, which is a different recursion rather than a
/// different family, so `Heston<_, _, AndersenQe>` stays on the host.
///
/// The variance's exponent travels as a parameter rather than picking a
/// family, since the kernel raises to it either way; whether the variance is
/// truncated or reflected does pick the family, as it does for every other
/// square-root diffusion here.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for Heston<T, S, Euler, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let pow_v = match self.pow {
      HestonPow::Sqrt => T::from_f64_fast(0.5),
      HestonPow::ThreeHalves => T::from_f64_fast(1.5),
    };
    if self.use_sym.unwrap_or(false) {
      crate::euler::EulerSpec::HestonReflected {
        mu: self.mu,
        kappa: self.kappa,
        theta: self.theta,
        sigma: self.sigma,
        rho: self.rho,
        pow_v,
      }
    } else {
      crate::euler::EulerSpec::Heston {
        mu: self.mu,
        kappa: self.kappa,
        theta: self.theta,
        sigma: self.sigma,
        rho: self.rho,
        pow_v,
      }
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.s0.unwrap_or(T::zero()),
      self.v0.unwrap_or(T::zero()).max(T::zero()),
      T::zero(),
      T::zero(),
    ]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The correlated-noise source divides the horizon by the number of points,
  /// not by the number of steps, so the device steps by that same amount.
  fn time_step(&self) -> T {
    self.cgns.dt()
  }

  fn device_seed(&self) -> u64 {
    rand::Rng::random(&mut self.seed.rng())
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt, Sch: HestonScheme] Heston<T, S, Sch> { s0, v0, kappa, theta, sigma, rho, mu, n, t, pow, use_sym, seed, cgns, _scheme } via euler);

/// Derives a seed once, at construction, for [`HestonSampler`] to own and
/// pass into `Sch::simulate` — see that trait method's docs for why.
/// Deriving (not cloning) is what decorrelates chunks: the derived value
/// is `self.seed`'s *mixed* next tick, not a raw snapshot, so chunk `i`'s
/// basis and chunk `i+1`'s basis are hash-scrambled relative to each
/// other rather than one raw stride apart.
macro_rules! heston_sampler_impl {
  ($scheme:ty) => {
    fn sampler(&self) -> HestonSampler<'_, T, S, $scheme, B> {
      HestonSampler {
        model: self,
        seed: self.seed.derive(),
      }
    }
  };
}

/// The Euler scheme is the one the engine reproduces, so this half of the
/// process routes through the backend.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for Heston<T, S, Euler, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = HestonSampler<'s, T, S, Euler, B>
  where
    Self: 's;

  heston_sampler_impl!(Euler);

  /// Through the Euler engine: on a device both components step in the
  /// kernel, on the host devices it is this process's own scheme, chunked
  /// exactly as `ProcessExt` chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    self.backend.system_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    self.backend.system_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    self.backend.system_paths(self, m)
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    self.backend.try_system_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// The quadratic-exponential scheme draws its variance from a non-central
/// chi-square rather than stepping an Euler recursion, so it has no family.
/// Its bound is `HostBackend`, which makes putting it on a device a compile
/// error at the first `sample` rather than a silent fall back to this code.
impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for Heston<T, S, AndersenQe, B> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = HestonSampler<'s, T, S, AndersenQe, B>
  where
    Self: 's;

  heston_sampler_impl!(AndersenQe);
}

/// Reusable [`Heston`] sampler: borrows the process and owns a seed derived
/// once at construction. The variance discretisation runs inside the
/// compile-time-selected [`HestonScheme`], which owns its own RNG setup
/// beyond that seed, so each call re-dispatches to `Sch::simulate`; there is
/// nothing else reusable to hoist across calls.
#[doc(hidden)]
pub struct HestonSampler<'a, T: FloatExt, S: SeedExt, Sch: HestonScheme, B> {
  model: &'a Heston<T, S, Sch, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, Sch: HestonScheme, B: Send + Sync> PathSampler<T>
  for HestonSampler<'_, T, S, Sch, B>
{
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    *out = Sch::simulate(self.model, &self.seed);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    Sch::simulate(self.model, &self.seed)
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> Heston<T, S, Euler, B> {
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
      Unseeded,
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
      Unseeded,
    );
    let [_s, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
  }

  /// Andersen QE: variance stays non-negative even with the Feller condition
  /// violated (2κθ = 0.16 < ξ² = 0.25), the simulated E[V_T] matches the exact
  /// CIR mean θ + (v0−θ)e^{−κT}, and the driftless asset is a martingale,
  /// E[S_T] ≈ S_0. Pinned seed; tolerances cover the MC error plus the small
  /// uncorrected-martingale bias of the plain QE asset scheme (§4.3 of
  /// Andersen has an optional exact correction not applied here).
  #[test]
  fn qe_variance_mean_and_asset_martingale() {
    use stochastic_rs_core::simd_rng::Deterministic;
    let (s0, v0, kappa, theta, sigma, rho, mu) = (100.0_f64, 0.04, 2.0, 0.04, 0.5, -0.7, 0.0);
    let (n, t, m) = (64usize, 1.0_f64, 30_000usize);
    let model = Heston::new(
      Some(s0),
      Some(v0),
      kappa,
      theta,
      sigma,
      rho,
      mu,
      n,
      Some(t),
      HestonPow::Sqrt,
      Some(false),
      Deterministic::new(20_240_601),
    )
    .qe();

    let mut sum_s = 0.0;
    let mut sum_v = 0.0;
    let mut nonneg = true;
    for _ in 0..m {
      let [s, v] = model.sample();
      sum_s += s[n - 1];
      sum_v += v[n - 1];
      if v.iter().any(|x| *x < 0.0) {
        nonneg = false;
      }
    }
    let mean_s = sum_s / m as f64;
    let mean_v = sum_v / m as f64;
    let v_exact = theta + (v0 - theta) * (-kappa * t).exp();

    assert!(
      nonneg,
      "QE variance must stay non-negative (Feller violated here)"
    );
    assert!(
      (mean_v - v_exact).abs() / v_exact < 0.05,
      "QE E[V_T] = {mean_v}, exact CIR mean = {v_exact}"
    );
    assert!(
      (mean_s - s0).abs() / s0 < 0.025,
      "QE asset not ~martingale: E[S_T] = {mean_s}, S_0 = {s0}"
    );
  }

  /// QE is a square-root (CIR) scheme; it must reject the 3/2 variance.
  #[test]
  #[should_panic(expected = "square-root (CIR) variance")]
  fn qe_rejects_three_halves() {
    let _ = Heston::new(
      Some(100.0_f64),
      Some(0.04),
      2.0,
      0.04,
      0.5,
      -0.7,
      0.0,
      16,
      Some(1.0),
      HestonPow::ThreeHalves,
      Some(false),
      Unseeded,
    )
    .qe()
    .sample();
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyHeston {
  inner_f32: Option<Heston<f32>>,
  inner_f64: Option<Heston<f64>>,
  seeded_f32: Option<Heston<f32, crate::simd_rng::Deterministic>>,
  seeded_f64: Option<Heston<f64, crate::simd_rng::Deterministic>>,
  /// The device the class samples on, chosen at construction.
  device: crate::python_device::Device,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHeston {
  #[new]
  #[pyo3(signature = (kappa, theta, sigma, rho, mu, n, s0=None, v0=None, t=None, pow=None, use_sym=None, seed=None, dtype=None, device=None))]
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
    seed: Option<u64>,
    dtype: Option<&str>,
    device: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    let device = crate::python_device::Device::parse(device, dtype.unwrap_or("f64"))?;
    let hp = match pow.unwrap_or("sqrt") {
      "three_halves" | "3/2" => HestonPow::ThreeHalves,
      _ => HestonPow::Sqrt,
    };
    let mut s = Self {
      inner_f32: None,
      inner_f64: None,
      seeded_f32: None,
      seeded_f64: None,
      device,
    };
    match (seed, dtype.unwrap_or("f64")) {
      (Some(sd), "f32") => {
        s.seeded_f32 = Some(Heston::new(
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
          Deterministic::new(sd),
        ));
      }
      (Some(sd), _) => {
        s.seeded_f64 = Some(Heston::new(
          s0,
          v0,
          kappa,
          theta,
          sigma,
          rho,
          mu,
          n,
          t,
          hp,
          use_sym,
          Deterministic::new(sd),
        ));
      }
      (None, "f32") => {
        s.inner_f32 = Some(Heston::new(
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
          Unseeded,
        ));
      }
      (None, _) => {
        s.inner_f64 = Some(Heston::new(
          s0, v0, kappa, theta, sigma, rho, mu, n, t, hp, use_sym, Unseeded,
        ));
      }
    }
    Ok(s)
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_device_dispatch!(self, |inner| {
      let [a, b] = inner.sample();
      (
        a.into_pyarray(py).into_py_any(py).unwrap(),
        b.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
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
    py_device_dispatch!(self, |inner| {
      let samples = inner.sample_par(m);
      let n = samples[0][0].len();
      let mut r0 = Array2::zeros((m, n));
      let mut r1 = Array2::zeros((m, n));
      for (i, [a, b]) in samples.iter().enumerate() {
        r0.row_mut(i).assign(a);
        r1.row_mut(i).assign(b);
      }
      (
        r0.into_pyarray(py).into_py_any(py).unwrap(),
        r1.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }
}
