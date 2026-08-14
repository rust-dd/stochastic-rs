//! # Non Central Chi Squared
//!
//! $$
//! X\sim\chi^2_\nu(\lambda),\quad f_X(x)=\tfrac12 e^{-(x+\lambda)/2}(x/\lambda)^{\nu/4-1/2}I_{\nu/2-1}(\sqrt{\lambda x})
//! $$
//!
//! Reference: Johnson, Kotz & Balakrishnan (1995), *Continuous Univariate
//! Distributions* vol. 2, §29.2 — decomposition
//! $\chi^2_\nu(\lambda) = \chi^2_{\nu-1} + (Z + \sqrt{\lambda})^2$ for $\nu \ge 1$,
//! and §29.4 — Poisson mixture
//! $\chi^2_\nu(\lambda) = \mathrm{Gamma}(\nu/2 + J,\ 2)$ with
//! $J \sim \mathrm{Poisson}(\lambda/2)$, valid for every $\nu > 0$.
use std::cell::Cell;

use stochastic_rs_core::simd_rng::SeedExt;

use crate::chi_square::SimdChiSquared;
use crate::gamma::SimdGamma;
use crate::normal::SimdNormal;
use crate::poisson::SimdPoisson;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::traits::FloatExt;
use crate::traits::SimdFloatExt;

/// Exact Poisson-mixture draw (module header §29.4): `χ²_df(λ) =
/// Gamma(df/2 + J, 2)` with `J ~ Poisson(λ/2)`, valid for every `df > 0`
/// but the only option when `0 < df < 1`, where the Gaussian-shift
/// decomposition (§29.2) does not exist — it needs a nonnegative central
/// χ²_{df−1} degrees of freedom. Shared by the free [`sample`] function and
/// [`SimdNonCentralChiSquared::sample_ncp`] so the two entry points cannot
/// diverge again the way they previously did.
#[inline]
fn poisson_mixture_sample<T: SimdFloatExt, S: SeedExt>(df: T, lambda: T, seed: &S) -> T {
  let two = T::from_f64_fast(2.0);
  let half_lambda = (lambda / two).to_f64().unwrap_or(f64::NAN);
  let mixture_jumps = if half_lambda > 0.0 {
    SimdPoisson::<u64>::new(half_lambda, seed).sample_fast()
  } else {
    0
  };
  let shape = df / two + T::from_f64_fast(mixture_jumps as f64);
  SimdGamma::<T>::new(shape, two, seed).sample_fast()
}

/// Stateful noncentral chi-squared sampler: `df` is fixed at construction,
/// the noncentrality parameter is passed per draw.
///
/// For `df ≥ 1`, the noncentrality enters only as a shift of the Gaussian
/// term in the decomposition above, so both sub-samplers (standard normal,
/// central χ²_{df−1}) stay buffered across draws. For `0 < df < 1`, where
/// that decomposition does not exist, [`Self::sample_ncp`] instead falls
/// back per draw to the crate-private `poisson_mixture_sample` helper —
/// the same branch the free [`sample`] function uses — reseeded from an
/// internal fork cursor so consecutive draws don't replay; buffering does
/// not help that branch since its Gamma shape depends on a fresh per-draw
/// Poisson jump count.
/// Use this struct over the one-shot [`sample`] free function in per-step
/// loops (e.g. exact Cir transitions, where `ncp` depends on the previous
/// state); for isolated `df < 1` draws the two cost about the same, since
/// neither buffers anything in that regime.
pub struct SimdNonCentralChiSquared<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  df: T,
  normal: SimdNormal<T, 64, R>,
  chisq: Option<SimdChiSquared<T, R>>,
  stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdNonCentralChiSquared<T, R> {
  /// Creates a sampler for χ²_df(·).
  ///
  /// - `df` — degrees of freedom (the module header's own ν), fixed at
  ///   construction. The noncentrality λ is **not** a constructor
  ///   argument — it is supplied per draw to [`Self::sample_ncp`].
  ///
  /// For `df ≥ 1` this also builds the Gaussian-shift decomposition's
  /// sub-samplers (dropping the central χ²_{df−1} term when `df ≈ 1`); for
  /// `0 < df < 1` those sub-samplers go unused by [`Self::sample_ncp`],
  /// which instead reseeds a fresh Poisson-mixture draw every call.
  pub fn new<S: SeedExt>(df: T, seed: &S) -> Self {
    let rem = df - T::one();
    let normal = SimdNormal::<T, 64, R>::new(T::zero(), T::one(), seed);
    let chisq = (rem > T::from_f64_fast(1e-10)).then(|| SimdChiSquared::<T, R>::new(rem, seed));
    // No own engine to seed for the df<1 Poisson-mixture branch — reuse
    // normal's already-captured stream_seed as this sampler's fork anchor
    // (see SimdChiSquared::new for the same pattern), so repeated df<1
    // draws advance instead of replaying the same sub-stream.
    let stream_seed = Cell::new(normal.stream_seed.get());
    Self {
      df,
      normal,
      chisq,
      stream_seed,
    }
  }

  /// Draws one χ²_df(ncp) sample.
  ///
  /// `ncp` must be non-negative — it is a noncentrality parameter, defined
  /// as a sum of squared means, so it cannot be negative in any valid use.
  /// This is not validated here: `ncp < 0.0` makes `ncp.sqrt()` `NaN`,
  /// which poisons the returned sample. The one in-tree caller
  /// (`stochastic-rs-stochastic`'s `volatility::svcgmy` Cir-style
  /// transition, in a sibling crate this one does not depend on and so
  /// cannot link to) passes a provably non-negative `ncp` (a sum of
  /// nonnegative terms), so this has not been an issue in practice — but
  /// the precondition is on the caller, not enforced here.
  #[inline]
  pub fn sample_ncp(&self, ncp: T) -> T {
    if self.df < T::one() {
      let mut basis = self.stream_seed.get();
      let child_seed = crate::simd_rng::derive_seed(&mut basis);
      self.stream_seed.set(basis);
      return poisson_mixture_sample(
        self.df,
        ncp,
        &crate::simd_rng::Deterministic::new(child_seed),
      );
    }
    let z = self.normal.sample_fast() + ncp.sqrt();
    let sq = z * z;
    match &self.chisq {
      Some(chisq) => chisq.sample_fast() + sq,
      None => sq,
    }
  }
}

/// One-shot noncentral chi-squared draw, exact for every `df > 0`.
///
/// For `df ≥ 1` this uses the Gaussian-shift decomposition of
/// [`SimdNonCentralChiSquared`]; for `0 < df < 1`, where that decomposition
/// does not exist, it falls back to the exact Poisson mixture
/// `χ²_df(λ) = Gamma(df/2 + J, 2)` with `J ~ Poisson(λ/2)`. Constructs the
/// sub-samplers per call — for repeated `df ≥ 1` draws hold a
/// [`SimdNonCentralChiSquared`] instead.
pub fn sample<T: FloatExt, S: SeedExt>(df: T, lambda: T, seed: &S) -> T {
  if df >= T::one() {
    return SimdNonCentralChiSquared::<T>::new(df, seed).sample_ncp(lambda);
  }
  poisson_mixture_sample(df, lambda, seed)
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::*;

  /// Backs `sample_ncp`'s own doc comment: a negative `ncp` is documented
  /// to poison the draw with `NaN` via `ncp.sqrt()`, unvalidated.
  #[test]
  fn sample_ncp_negative_is_nan() {
    let s = SimdNonCentralChiSquared::<f64>::new(3.0, &Unseeded);
    assert!(s.sample_ncp(-1.0).is_nan());
  }

  /// Non-negative `ncp` (including exactly zero) must stay finite.
  #[test]
  fn sample_ncp_nonnegative_is_finite() {
    let s = SimdNonCentralChiSquared::<f64>::new(3.0, &Unseeded);
    assert!(s.sample_ncp(0.0).is_finite());
    assert!(s.sample_ncp(2.5).is_finite());
  }

  /// The struct path must agree with the free [`sample`] function for
  /// `0 < df < 1`: mean = `df + lambda`, variance = `2*(df + 2*lambda)`
  /// (Johnson, Kotz & Balakrishnan §29.4 cumulants). Before the fix,
  /// `sample_ncp` silently treated any `df` in this range as `df ≈ 1`
  /// (dropping it and sampling `(Z + sqrt(lambda))^2`), which has mean
  /// `1 + lambda` and variance `2 + 4*lambda` instead — for `df = 0.3`,
  /// `lambda = 2.0` that wrong-path mean/variance (3.0 / 10.0) sit tens of
  /// standard errors away from the correct closed form (2.3 / 8.6) at the
  /// sample size below, so this test fails loudly against the old path
  /// rather than by a coin-flip margin.
  #[test]
  fn sample_ncp_df_below_one_matches_closed_form_moments() {
    let df = 0.3_f64;
    let lambda = 2.0_f64;
    let n = 100_000_usize;
    let dist = SimdNonCentralChiSquared::<f64>::new(df, &Deterministic::new(11));
    let samples = (0..n).map(|_| dist.sample_ncp(lambda)).collect::<Vec<_>>();

    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    let expected_mean = df + lambda;
    let expected_var = 2.0 * (df + 2.0 * lambda);

    // Exact cumulant formula kappa_r = 2^(r-1) (r-1)! (df + r*lambda) gives
    // the noncentral chi-squared's 4th cumulant, hence its 4th central
    // moment mu4 = kappa4 + 3*kappa2^2, hence the large-n variance of the
    // plug-in variance estimator: Var(S^2) ~= (mu4 - kappa2^2) / n.
    let se_mean = (expected_var / n as f64).sqrt();
    let kappa4 = 48.0 * (df + 4.0 * lambda);
    let mu4 = kappa4 + 3.0 * expected_var * expected_var;
    let se_var = ((mu4 - expected_var * expected_var) / n as f64).sqrt();

    assert!(
      (mean - expected_mean).abs() < 6.0 * se_mean,
      "mean {mean} vs expected {expected_mean} (6*SE = {})",
      6.0 * se_mean
    );
    assert!(
      (var - expected_var).abs() < 6.0 * se_var,
      "variance {var} vs expected {expected_var} (6*SE = {})",
      6.0 * se_var
    );
  }
}
