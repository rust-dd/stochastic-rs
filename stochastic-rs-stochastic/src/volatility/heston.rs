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
use stochastic_rs_distributions::normal::SimdNormal;

use super::HestonPow;
use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Compile-time selector for the variance-discretisation scheme that
/// [`Heston::sample`] runs. The schemes are zero-sized marker types and the
/// choice is a type parameter, so each variant is monomorphised separately:
/// the default [`Euler`] path keeps its exact code generation (no runtime
/// branch on the scheme), and [`AndersenQe`] is a wholly independent code
/// path selected at compile time via [`Heston::qe`].
pub trait HestonScheme: Send + Sync + 'static {
  /// Generate `[stock path, variance path]` under this scheme.
  fn simulate<T: FloatExt, S: SeedExt>(model: &Heston<T, S, Self>) -> [Array1<T>; 2]
  where
    Self: Sized;
}

/// Full-truncation (or reflection, when `use_sym`) Euler–Maruyama — the
/// original Heston discretisation. Default scheme; behaviour is unchanged.
pub struct Euler;

/// Andersen (2008) Quadratic-Exponential scheme. Markedly lower variance bias
/// than Euler at large vol-of-vol / when the Feller condition is violated, at
/// essentially the same per-step cost (no Cholesky, correlation handled
/// analytically). Defined for the square-root (CIR) variance only.
///
/// Reference: Andersen, L. (2008), "Simple and efficient simulation of the
/// Heston stochastic volatility model", *Journal of Computational Finance*
/// 11(3), 1-42 (§3.2 scheme QE, §4.2 eq. 33 for the asset).
pub struct AndersenQe;

pub struct Heston<T: FloatExt, S: SeedExt = Unseeded, Sch: HestonScheme = Euler> {
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
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  /// Noise generator (used by the [`Euler`] scheme; [`AndersenQe`] draws its
  /// own noise and leaves this untouched).
  cgns: Cgns<T>,
  /// Zero-sized marker for the compile-time variance scheme.
  _scheme: PhantomData<Sch>,
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

  /// Switch to the [`AndersenQe`] variance scheme at compile time. Consumes
  /// the model and re-tags it — zero runtime cost (the fields are moved and
  /// the marker swapped). QE is defined for the square-root (CIR) variance,
  /// so keep `pow = HestonPow::Sqrt`.
  pub fn qe(self) -> Heston<T, S, AndersenQe> {
    Heston {
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

impl<T: FloatExt, S: SeedExt, Sch: HestonScheme> ProcessExt<T> for Heston<T, S, Sch> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = HestonSampler<'s, T, S, Sch>
  where
    Self: 's;

  fn sampler(&self) -> HestonSampler<'_, T, S, Sch> {
    HestonSampler(self)
  }
}

/// Borrow-based [`Heston`] sampler. The variance discretisation runs inside the
/// compile-time-selected [`HestonScheme`], which owns its own RNG setup, so
/// each call re-dispatches to `Sch::simulate`; there is nothing reusable to
/// hoist across calls.
#[doc(hidden)]
pub struct HestonSampler<'a, T: FloatExt, S: SeedExt, Sch: HestonScheme>(&'a Heston<T, S, Sch>);

impl<T: FloatExt, S: SeedExt, Sch: HestonScheme> PathSampler<T> for HestonSampler<'_, T, S, Sch> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    *out = Sch::simulate(self.0);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    Sch::simulate(self.0)
  }
}

impl HestonScheme for Euler {
  fn simulate<T: FloatExt, S: SeedExt>(model: &Heston<T, S, Euler>) -> [Array1<T>; 2] {
    let dt = model.cgns.dt();
    let [cgn1, cgn2] = &model.cgns.sample_impl(&model.seed.derive());

    let mut s = Array1::<T>::zeros(model.n);
    let mut v = Array1::<T>::zeros(model.n);

    s[0] = model.s0.unwrap_or(T::zero());
    v[0] = model.v0.unwrap_or(T::zero()).max(T::zero());

    for i in 1..model.n {
      let v_prev = v[i - 1].max(T::zero());
      s[i] = s[i - 1] + model.mu * s[i - 1] * dt + s[i - 1] * v_prev.sqrt() * cgn1[i - 1];

      let dv = model.kappa * (model.theta - v_prev) * dt
        + model.sigma
          * v_prev.powf(match model.pow {
            HestonPow::Sqrt => T::from_f64_fast(0.5),
            HestonPow::ThreeHalves => T::from_f64_fast(1.5),
          })
          * cgn2[i - 1];

      v[i] = match model.use_sym.unwrap_or(false) {
        true => (v[i - 1] + dv).abs(),
        false => (v[i - 1] + dv).max(T::zero()),
      }
    }

    [s, v]
  }
}

impl HestonScheme for AndersenQe {
  /// Andersen (2008) QE step. Per step: one variance draw — quadratic branch
  /// `V = a(b+Z_V)²` (eq. 23/27/28) when `ψ ≤ ψ_c`, else exponential
  /// `V = Ψ⁻¹(U)` (eq. 24-26/29/30) — followed by the asset update (eq. 33).
  /// Correlation is handled analytically through the `K` constants, so no
  /// correlated Brownian pair is needed (unlike [`Euler`]).
  fn simulate<T: FloatExt, S: SeedExt>(model: &Heston<T, S, AndersenQe>) -> [Array1<T>; 2] {
    assert!(
      matches!(model.pow, HestonPow::Sqrt),
      "Andersen QE is defined only for the square-root (CIR) variance; use HestonPow::Sqrt"
    );
    assert!(
      model.kappa > T::zero(),
      "Andersen QE requires a positive mean-reversion rate kappa"
    );

    let n = model.n;
    let dt = model.t.unwrap_or(T::one()) / T::from_usize_(n - 1);

    let kappa = model.kappa;
    let theta = model.theta;
    let eps = model.sigma; // vol-of-vol ε
    let rho = model.rho;
    let mu = model.mu;

    let one = T::one();
    let two = T::from_f64_fast(2.0);
    let half = T::from_f64_fast(0.5);
    let psi_c = T::from_f64_fast(1.5);

    // Time-independent constants, hoisted out of the path loop (Andersen §3
    // fn. 5). Central discretisation of ∫V du uses γ₁ = γ₂ = ½ (eq. 33), so
    // K₃ = K₄ = ½Δ(1−ρ²).
    let e_kd = (-kappa * dt).exp(); // e^{−κΔ}
    let krho_eps = kappa * rho / eps;
    let k0 = -rho * kappa * theta * dt / eps;
    let k1 = half * dt * (krho_eps - half) - rho / eps;
    let k2 = half * dt * (krho_eps - half) + rho / eps;
    let k34 = half * dt * (one - rho * rho);

    let s0 = model.s0.unwrap_or(T::one());
    let v0 = model.v0.unwrap_or(T::zero()).max(T::zero());
    assert!(
      s0 > T::zero(),
      "Andersen QE evolves log-spot, so s0 must be > 0"
    );

    let mut s = Array1::<T>::zeros(n);
    let mut v = Array1::<T>::zeros(n);
    s[0] = s0;
    v[0] = v0;

    // Independent noise sub-streams: normals (Z_V for the quadratic branch and
    // Z for the asset) via the buffered SimdNormal, a uniform stream for the
    // exponential branch. Built here because SimdNormal is not `Sync`.
    let normal = SimdNormal::<T>::new(T::zero(), T::one(), &model.seed.derive());
    let mut urng = model.seed.derive().rng();

    let mut log_s = s0.ln();
    let mut v_prev = v0;
    for i in 1..n {
      // Conditional moments of V_i given V_{i−1} (eq. 17, 18).
      let m = theta + (v_prev - theta) * e_kd;
      let s2 = v_prev * eps * eps * e_kd / kappa * (one - e_kd)
        + theta * eps * eps / (two * kappa) * (one - e_kd) * (one - e_kd);
      let psi = s2 / (m * m);

      let v_next = if psi <= psi_c {
        // Quadratic branch (eq. 27, 28, 23).
        let inv = two / psi; // 2ψ⁻¹
        let b2 = inv - one + (inv * (inv - one)).sqrt();
        let a = m / (one + b2);
        let b = b2.sqrt();
        let zv = normal.sample_fast();
        a * (b + zv) * (b + zv)
      } else {
        // Exponential branch (eq. 29, 30, 25): mass p at 0 + exponential tail.
        let p = (psi - one) / (psi + one);
        let beta = (one - p) / m; // = 2 / (m(ψ+1))
        let u = T::sample_uniform_simd(&mut urng);
        if u <= p {
          T::zero()
        } else {
          ((one - p) / (one - u)).ln() / beta
        }
      };

      // Asset (eq. 33). The real drift μΔ is added on top of the QE
      // correlation/Itô constants; Z is independent of V_next.
      let z = normal.sample_fast();
      let vol = (k34 * (v_prev + v_next)).max(T::zero()).sqrt();
      log_s = log_s + mu * dt + k0 + k1 * v_prev + k2 * v_next + vol * z;

      v[i] = v_next;
      s[i] = log_s.exp();
      v_prev = v_next;
    }

    [s, v]
  }
}

impl<T: FloatExt, S: SeedExt> Heston<T, S> {
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
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHeston {
  #[new]
  #[pyo3(signature = (kappa, theta, sigma, rho, mu, n, s0=None, v0=None, t=None, pow=None, use_sym=None, seed=None, dtype=None))]
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
  ) -> Self {
    let hp = match pow.unwrap_or("sqrt") {
      "three_halves" | "3/2" => HestonPow::ThreeHalves,
      _ => HestonPow::Sqrt,
    };
    let mut s = Self {
      inner_f32: None,
      inner_f64: None,
      seeded_f32: None,
      seeded_f64: None,
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
    s
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
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
    py_dispatch!(self, |inner| {
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
