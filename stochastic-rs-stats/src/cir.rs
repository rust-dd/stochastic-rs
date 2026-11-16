//! # Cir
//!
//! $$
//! dX_t=\theta(\mu-X_t)\,dt+\sigma\sqrt{X_t}\,dW_t
//! $$
//!
//! Exact transition sampling and density via the scaled noncentral chi-squared
//! law of the process.
//!
//! Reference: Cox, J.C., Ingersoll, J.E., Ross, S.A. (1985), "A Theory of the
//! Term Structure of Interest Rates", *Econometrica* 53(2), 385-407.
//! DOI: 10.2307/1911242
//!
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::params::VarianceBoundary;

/// Samples one exact Cox-Ingersoll-Ross transition with an entropy-seeded RNG.
///
/// `theta` is the mean-reversion speed and `mu` is the long-run mean. Invalid
/// or non-finite inputs return `NaN`; at `t == 0` the current state is returned.
/// A `sigma` so small that the χ² parameters overflow returns the conditional mean (relative spread < 1e-154).
pub fn sample(theta: f64, mu: f64, sigma: f64, t: f64, r_t: f64) -> f64 {
  law(theta, mu, sigma).map_or(f64::NAN, |cir| cir.sample_transition(t, r_t, &Unseeded))
}

/// Samples one reproducible exact Cox-Ingersoll-Ross transition.
pub fn sample_seeded(theta: f64, mu: f64, sigma: f64, t: f64, r_t: f64, seed: u64) -> f64 {
  law(theta, mu, sigma).map_or(f64::NAN, |cir| {
    cir.sample_transition(t, r_t, &Deterministic::new(seed))
  })
}

/// Exact Cox-Ingersoll-Ross transition density.
///
/// Returns NaN for parameters outside the valid domain, at `t == 0`, and for a
/// negative or non-finite `future_state`.
pub fn pdf(theta: f64, mu: f64, sigma: f64, t: f64, r_t: f64, future_state: f64) -> f64 {
  law(theta, mu, sigma).map_or(f64::NAN, |cir| cir.transition_pdf(t, r_t, future_state))
}

/// Cox-Ingersoll-Ross (Cir) process Asymptotic PDF.
pub fn apdf(theta: f64, mu: f64, sigma: f64, r_t: f64) -> f64 {
  law(theta, mu, sigma).map_or(f64::NAN, |cir| cir.stationary_pdf(r_t))
}

/// `None`, so NaN, for a non-finite parameter, which `Cir::new` refuses; the law reads no boundary,
/// and `Reflect` keeps the floored path's sub-Feller warning out of every evaluation.
fn law(theta: f64, mu: f64, sigma: f64) -> Option<Cir<f64>> {
  (theta.is_finite() && mu.is_finite() && sigma.is_finite()).then(|| {
    Cir::new(
      theta,
      mu,
      sigma,
      2,
      None,
      None,
      VarianceBoundary::Reflect,
      Unseeded,
    )
  })
}

#[cfg(test)]
#[path = "cir_tests.rs"]
mod tests;
