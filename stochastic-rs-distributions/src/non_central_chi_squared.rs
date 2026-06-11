//! # Non Central Chi Squared
//!
//! $$
//! X\sim\chi^2_\nu(\lambda),\quad f_X(x)=\tfrac12 e^{-(x+\lambda)/2}(x/\lambda)^{\nu/4-1/2}I_{\nu/2-1}(\sqrt{\lambda x})
//! $$
//!
//! Reference: Johnson, Kotz & Balakrishnan (1995), *Continuous Univariate
//! Distributions* vol. 2, §29.2 — decomposition
//! $\chi^2_\nu(\lambda) = \chi^2_{\nu-1} + (Z + \sqrt{\lambda})^2$ for $\nu \ge 1$.
use stochastic_rs_core::simd_rng::SeedExt;

use crate::chi_square::SimdChiSquared;
use crate::normal::SimdNormal;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::traits::FloatExt;
use crate::traits::SimdFloatExt;

/// Stateful noncentral chi-squared sampler: `df` is fixed at construction,
/// the noncentrality parameter is passed per draw.
///
/// The noncentrality enters only as a shift of the Gaussian term in the
/// decomposition above, so both sub-samplers (standard normal, central
/// χ²_{df−1}) stay buffered across draws. Use this in per-step loops (e.g.
/// exact Cir transitions, where `ncp` depends on the previous state) instead
/// of the one-shot [`sample`] free function, which rebuilds both samplers
/// and their RNGs on every call.
pub struct SimdNonCentralChiSquared<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  normal: SimdNormal<T, 64, R>,
  chisq: Option<SimdChiSquared<T, R>>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdNonCentralChiSquared<T, R> {
  /// Creates a sampler for χ²_df(·). The decomposition assumes `df ≥ 1`;
  /// the central χ²_{df−1} term is dropped when `df ≈ 1`.
  pub fn new<S: SeedExt>(df: T, seed: &S) -> Self {
    let rem = df - T::one();
    Self {
      normal: SimdNormal::<T, 64, R>::new(T::zero(), T::one(), seed),
      chisq: (rem > T::from_f64_fast(1e-10)).then(|| SimdChiSquared::<T, R>::new(rem, seed)),
    }
  }

  /// Draws one χ²_df(ncp) sample.
  #[inline]
  pub fn sample_ncp(&self, ncp: T) -> T {
    let z = self.normal.sample_fast() + ncp.sqrt();
    let sq = z * z;
    match &self.chisq {
      Some(chisq) => chisq.sample_fast() + sq,
      None => sq,
    }
  }
}

/// One-shot convenience wrapper around [`SimdNonCentralChiSquared`].
///
/// Constructs the sampler (two RNG seedings + buffer fills) per call — for
/// repeated draws hold a [`SimdNonCentralChiSquared`] instead.
pub fn sample<T: FloatExt, S: SeedExt>(df: T, lambda: T, seed: &S) -> T {
  SimdNonCentralChiSquared::<T>::new(df, seed).sample_ncp(lambda)
}
