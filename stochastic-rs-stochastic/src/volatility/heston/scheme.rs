//! Compile-time variance-discretisation schemes for [`Heston`](super::Heston): [`Euler`], the
//! full-truncation scheme, and [`AndersenQe`], Andersen 2008's Quadratic-Exponential scheme.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::Heston;
use crate::params::VarianceBoundary;
use crate::traits::FloatExt;
use crate::volatility::HestonPow;

mod sealed {
  #[diagnostic::on_unimplemented(
    message = "`{Self}` cannot implement the sealed trait `HestonScheme`",
    note = "the variance schemes are `Euler` and `AndersenQe`"
  )]
  pub trait Sealed {}
}

impl sealed::Sealed for Euler {}

impl sealed::Sealed for AndersenQe {}

/// Compile-time selector of the variance discretisation [`Heston`](super::Heston) runs: a
/// zero-sized marker per scheme, monomorphised on its own; sealed to [`Euler`] and [`AndersenQe`].
pub trait HestonScheme: sealed::Sealed + Send + Sync + 'static {
  /// `[stock path, variance path]` under this scheme, drawing from `seed`: the basis the calling
  /// [`HestonSampler`](super::HestonSampler) owns, never `model.seed`, so chunks do not race on it.
  fn simulate<T: FloatExt, S: SeedExt, B>(
    model: &Heston<T, Self, S, B>,
    seed: &S,
  ) -> [Array1<T>; 2]
  where
    Self: Sized;
}

/// Full-truncation Euler–Maruyama, or reflection under [`VarianceBoundary::Reflect`]: the original
/// Heston discretisation and the default scheme.
#[derive(Debug, Clone, Copy)]
pub struct Euler;

/// Andersen's Quadratic-Exponential scheme, for the square-root variance only: lower bias than
/// Euler at large vol-of-vol or a violated Feller condition, at about the same cost per step.
/// Reference: Andersen (2008), 11(3), 1-42, §3.2 (QE), §4.2 eq. 33, DOI 10.21314/JCF.2008.189
#[derive(Debug, Clone, Copy)]
pub struct AndersenQe;

impl HestonScheme for Euler {
  fn simulate<T: FloatExt, S: SeedExt, B>(
    model: &Heston<T, Euler, S, B>,
    seed: &S,
  ) -> [Array1<T>; 2] {
    let dt = model.cgns.dt();
    let [cgn1, cgn2] = &model.cgns.sample_impl(seed);

    let mut s = Array1::<T>::zeros(model.n);
    let mut v = Array1::<T>::zeros(model.n);

    s[0] = model.s0;
    v[0] = model.v0.max(T::zero());

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

      v[i] = match model.boundary {
        VarianceBoundary::Floor => (v[i - 1] + dv).max(T::zero()),
        VarianceBoundary::Reflect => (v[i - 1] + dv).abs(),
      }
    }

    [s, v]
  }
}

/// The step keeps the variance non-negative by construction, so it does not read `boundary`.
impl HestonScheme for AndersenQe {
  /// Andersen (2008) QE step: `V = a(b+Z_V)²` (eq. 23/27/28) when `ψ ≤ ψ_c`, else `V = Ψ⁻¹(U)`
  /// (eq. 24-26/29/30), then the asset (eq. 33); the `K` constants carry the correlation.
  fn simulate<T: FloatExt, S: SeedExt, B>(
    model: &Heston<T, AndersenQe, S, B>,
    seed: &S,
  ) -> [Array1<T>; 2] {
    model.assert_qe_parameters();

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

    // Time-independent constants, hoisted out of the path loop (Andersen §3 fn. 5). The central
    // discretisation of ∫V du uses γ₁ = γ₂ = ½ (eq. 33), so K₃ = K₄ = ½Δ(1−ρ²).
    let e_kd = (-kappa * dt).exp(); // e^{−κΔ}
    let krho_eps = kappa * rho / eps;
    let k0 = -rho * kappa * theta * dt / eps;
    let k1 = half * dt * (krho_eps - half) - rho / eps;
    let k2 = half * dt * (krho_eps - half) + rho / eps;
    let k34 = half * dt * (one - rho * rho);

    let s0 = model.s0;
    let v0 = model.v0.max(T::zero());

    let mut s = Array1::<T>::zeros(n);
    let mut v = Array1::<T>::zeros(n);
    s[0] = s0;
    v[0] = v0;

    // Normals (Z_V and the asset's Z) and a uniform stream for the exponential branch. `normal`
    // consumes the sampler's owned basis; `urng` derives from it, which keeps the two independent.
    let mut normal = SimdNormal::<T>::new(T::zero(), T::one()).seeded(seed);
    let mut urng = seed.derive().rng();

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
        let zv = normal.sample();
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
      let z = normal.sample();
      let vol = (k34 * (v_prev + v_next)).max(T::zero()).sqrt();
      log_s = log_s + mu * dt + k0 + k1 * v_prev + k2 * v_next + vol * z;

      v[i] = v_next;
      s[i] = log_s.exp();
      v_prev = v_next;
    }

    [s, v]
  }
}
