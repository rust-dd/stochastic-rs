//! Compile-time variance-discretisation schemes for
//! [`Heston`](super::Heston): [`Euler`] (the original full-truncation
//! scheme) and [`AndersenQe`] (Andersen 2008's Quadratic-Exponential
//! scheme). Split out of `heston.rs` to keep that file under the crate's
//! 600-line cap; re-exported from `heston` so callers see no path change.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::normal::SimdNormal;

use super::Heston;
use crate::traits::FloatExt;
use crate::volatility::HestonPow;

/// Compile-time selector for the variance-discretisation scheme that
/// [`Heston::sample`](super::Heston) runs. The schemes are zero-sized
/// marker types and the choice is a type parameter, so each variant is
/// monomorphised separately:
/// the default [`Euler`] path keeps its exact code generation (no runtime
/// branch on the scheme), and [`AndersenQe`] is a wholly independent code
/// path selected at compile time via [`Heston::qe`].
pub trait HestonScheme: Send + Sync + 'static {
  /// Generate `[stock path, variance path]` under this scheme, drawing from
  /// `seed` — a basis owned by the calling
  /// [`HestonSampler`](super::HestonSampler), derived once at its
  /// construction, never `model.seed` directly. That indirection is what
  /// makes `sample_par`/`sample_map`'s chunked fan-out deterministic: each
  /// chunk's sampler owns a distinct basis instead of every chunk racing on
  /// the same live `model.seed` inside the parallel region.
  fn simulate<T: FloatExt, S: SeedExt>(model: &Heston<T, S, Self>, seed: &S) -> [Array1<T>; 2]
  where
    Self: Sized;
}

/// Full-truncation (or reflection, when `use_sym`) Euler–Maruyama — the
/// original Heston discretisation. Default scheme; behaviour is unchanged.
#[derive(Clone, Copy)]
pub struct Euler;

/// Andersen (2008) Quadratic-Exponential scheme. Markedly lower variance bias
/// than Euler at large vol-of-vol / when the Feller condition is violated, at
/// essentially the same per-step cost (no Cholesky, correlation handled
/// analytically). Defined for the square-root (CIR) variance only.
///
/// Reference: Andersen, L. (2008), "Simple and efficient simulation of the
/// Heston stochastic volatility model", *Journal of Computational Finance*
/// 11(3), 1-42 (§3.2 scheme QE, §4.2 eq. 33 for the asset).
#[derive(Clone, Copy)]
pub struct AndersenQe;

impl HestonScheme for Euler {
  fn simulate<T: FloatExt, S: SeedExt>(model: &Heston<T, S, Euler>, seed: &S) -> [Array1<T>; 2] {
    let dt = model.cgns.dt();
    let [cgn1, cgn2] = &model.cgns.sample_impl(seed);

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
  fn simulate<T: FloatExt, S: SeedExt>(
    model: &Heston<T, S, AndersenQe>,
    seed: &S,
  ) -> [Array1<T>; 2] {
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
    // exponential branch. Built here because SimdNormal is not `Sync`. `seed`
    // is `HestonSampler`'s own owned basis (already chunk-decorrelated by
    // `sampler()`'s one derive), so `normal` consumes it directly; `urng`
    // still derives *from* it — a second, within-chunk hop that keeps the
    // two sub-streams independent without affecting cross-chunk decorrelation,
    // since it operates entirely on an already-decorrelated basis.
    let normal = SimdNormal::<T>::new(T::zero(), T::one(), seed);
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
