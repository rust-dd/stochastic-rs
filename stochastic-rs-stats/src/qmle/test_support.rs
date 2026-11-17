use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;

/// Exact OU transition: Gaussian with mean μ+(X−μ)e^{−θΔ} and variance σ²(1−e^{−2θΔ})/(2θ).
pub(crate) fn simulate_ou(
  theta: f64,
  mu: f64,
  sigma: f64,
  x0: f64,
  dt: f64,
  n: usize,
  seed: u64,
) -> Array1<f64> {
  let a = (-theta * dt).exp();
  let sd = (sigma * sigma * (1.0 - a * a) / (2.0 * theta)).sqrt();
  let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
  let mut path = Array1::<f64>::zeros(n);
  path[0] = x0;
  for t in 1..n {
    let z = normal.sample();
    path[t] = mu + (path[t - 1] - mu) * a + sd * z;
  }
  path
}

/// Exact CIR step `c χ'²_d(λ)`: `c = σ²(1−a)/(4θ)`, `d = 4θμ/σ²`, `λ = 4θa X_t/(σ²(1−a))`, the
/// non-central χ² drawn as a Poisson mixture of central ones (Gammas).
pub(crate) fn simulate_cir(
  theta: f64,
  mu: f64,
  sigma: f64,
  x0: f64,
  dt: f64,
  n: usize,
  seed: u64,
) -> Array1<f64> {
  let a = (-theta * dt).exp();
  let s2 = sigma * sigma;
  let c = s2 * (1.0 - a) / (4.0 * theta);
  let d = 4.0 * theta * mu / s2;
  let mut path = Array1::<f64>::zeros(n);
  path[0] = x0;
  // λ and the χ² shape change every step, so each step seeds its own stream from `seed` and `t`:
  // one seed for every step would restart the same stream and destroy the path.
  for t in 1..n {
    let lambda = path[t - 1] * 4.0 * theta * a / (s2 * (1.0 - a));
    let pois_seed = seed ^ (t as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let gamma_seed = seed ^ (t as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let n_pois = f64::from(
      SimdPoisson::<u32>::new(lambda / 2.0)
        .seeded(&Deterministic::new(pois_seed))
        .sample(),
    );
    let shape = d / 2.0 + n_pois;
    let chi2 = if shape > 0.0 {
      SimdGamma::<f64>::new(shape, 2.0)
        .seeded(&Deterministic::new(gamma_seed))
        .sample()
    } else {
      0.0
    };
    path[t] = c * chi2;
  }
  path
}
