//! # Non Central Chi Squared
//!
//! $$
//! X\sim\chi^2_\nu(\lambda),\quad f_X(x)=\tfrac12 e^{-(x+\lambda)/2}(x/\lambda)^{\nu/4-1/2}I_{\nu/2-1}(\sqrt{\lambda x})
//! $$
//!
//! Sampling: the shift $\chi^2_\nu(\lambda) = \chi^2_{\nu-1} + (Z + \sqrt{\lambda})^2$ for $\nu \ge 1$, else the Poisson
//! mixture $\mathrm{Gamma}(\nu/2 + J,\ 2)$ with $J \sim \mathrm{Poisson}(\lambda/2)$, valid for every $\nu > 0$.
//!
//! - Johnson, N.L., Kotz, S., Balakrishnan, N. (1995), *Continuous Univariate Distributions*, vol. 2, 2nd ed., Wiley, ch. 29.
//! - Abramowitz, M., Stegun, I.A. (1964), *Handbook of Mathematical Functions*, NBS AMS 55, eq. 26.4.25 (the Poisson mixture).
use rand::Rng;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;
use stochastic_rs_core::simd_rng::derive_seed;

use crate::chi_square::SimdChiSquared;
use crate::gamma::GammaState;
use crate::gamma::SimdGamma;
use crate::normal::SimdNormal;
use crate::poisson::SimdPoisson;
use crate::seeded::Seeded;
use crate::seeded::StreamState;
use crate::source::AnyRng;
use crate::traits::FloatExt;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// The Poisson count and the gamma one Poisson-mixture draw consumes, from a seed or from the caller's rng.
trait MixtureSource<T: SimdFloatExt> {
  fn poisson(&mut self, law: SimdPoisson<u64>) -> u64;

  fn gamma(&mut self, law: SimdGamma<T>) -> T;
}

impl<T: SimdFloatExt, S: SeedExt> MixtureSource<T> for &S {
  fn poisson(&mut self, law: SimdPoisson<u64>) -> u64 {
    law.seeded(*self).sample()
  }

  fn gamma(&mut self, law: SimdGamma<T>) -> T {
    law.seeded(*self).sample()
  }
}

impl<T: SimdFloatExt, G: Rng + ?Sized> MixtureSource<T> for AnyRng<'_, G> {
  fn poisson(&mut self, law: SimdPoisson<u64>) -> u64 {
    law.draw_with(self.0)
  }

  fn gamma(&mut self, law: SimdGamma<T>) -> T {
    law.draw_with(self.0)
  }
}

/// The Poisson mixture `Gamma(df/2 + J, 2)` with `J ~ Poisson(ncp/2)`, the only exact form for `0 < df < 1`.
fn poisson_mixture<T: SimdFloatExt>(df: T, ncp: T, mut src: impl MixtureSource<T>) -> T {
  let two = T::from_f64_fast(2.0);
  let half_lambda = (ncp / two).to_f64().unwrap_or(f64::NAN);
  let mixture_jumps = if half_lambda > 0.0 {
    src.poisson(SimdPoisson::new(half_lambda))
  } else {
    0
  };
  let shape = df / two + T::from_f64_fast(mixture_jumps as f64);
  src.gamma(SimdGamma::new(shape, two))
}

/// Noncentral chi-squared law with `df` degrees of freedom whose noncentrality is a per-draw argument, so it has no
/// `Distribution`: a seeded stream draws it with `sample_ncp`, the caller's rng with [`Self::sample_ncp_with`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdNonCentralChiSquared<T> {
  df: T,
  chisq: Option<SimdChiSquared<T>>,
}

/// A noncentral chi-squared stream: the shift's normal, the central `χ²_{df−1}`, and the cursor that seeds each
/// `df < 1` mixture draw.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct NcxState<T: SimdFloatExt, R: SimdRngExt> {
  normal: StreamState<T, R, 64>,
  chisq: Option<GammaState<T, R>>,
  cursor: u64,
}

impl<T: SimdFloatExt> SimdNonCentralChiSquared<T> {
  /// The law of `χ²_df(·)` with `df` degrees of freedom ν > 0; the central `χ²_{df−1}` term is dropped when `df ≈ 1`.
  pub fn new(df: T) -> Self {
    assert!(
      df > T::zero(),
      "df must satisfy `df > T::zero()`, got df = {df:?}"
    );
    let rem = df - T::one();
    Self {
      df,
      chisq: (rem > T::from_f64_fast(1e-10)).then(|| SimdChiSquared::<T>::new(rem)),
    }
  }

  /// The degrees of freedom `ν`.
  pub fn df(&self) -> T {
    self.df
  }

  /// One draw of `χ²_df(ncp)` on the caller's rng: the shift for `df ≥ 1`, the Poisson mixture below.
  pub fn sample_ncp_with<G: Rng + ?Sized>(&self, rng: &mut G, ncp: T) -> T {
    if self.df < T::one() {
      return poisson_mixture(self.df, ncp, AnyRng(rng));
    }
    let z = SimdNormal::<T>::standard().draw_with(rng) + ncp.sqrt();
    let sq = z * z;
    match &self.chisq {
      Some(chisq) => chisq.draw_with(rng) + sq,
      None => sq,
    }
  }
}

impl<T: SimdFloatExt> Sealed for SimdNonCentralChiSquared<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdNonCentralChiSquared<T> {
  type State<R: SimdRngExt> = NcxState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (NcxState<T, R>, u64) {
    let (normal, basis) = SimdNormal::<T>::standard().init::<R, S>(seed);
    let chisq = self.chisq.map(|c| c.init::<R, S>(seed).0);
    (
      NcxState {
        normal,
        chisq,
        cursor: basis,
      },
      basis,
    )
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Seeded<SimdNonCentralChiSquared<T>, R> {
  /// One draw of `χ²_df(ncp)`; `ncp` must be non-negative (unchecked: a negative one gives `NaN` through `√ncp`).
  #[inline]
  pub fn sample_ncp(&mut self, ncp: T) -> T {
    let (law, state) = self.parts_mut();
    if law.df < T::one() {
      let child = Deterministic::new(derive_seed(&mut state.cursor));
      return poisson_mixture(law.df, ncp, &child);
    }
    let z = SimdNormal::<T>::standard().next(&mut state.normal) + ncp.sqrt();
    let sq = z * z;
    match (&law.chisq, &mut state.chisq) {
      (Some(chisq), Some(chisq_state)) => chisq.next(chisq_state) + sq,
      _ => sq,
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
    return SimdNonCentralChiSquared::<T>::new(df)
      .seeded(seed)
      .sample_ncp(lambda);
  }
  poisson_mixture(df, lambda, seed)
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::SimdRng;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::*;

  /// Backs `sample_ncp`'s own doc comment: a negative `ncp` is documented
  /// to poison the draw with `NaN` via `ncp.sqrt()`, unvalidated.
  #[test]
  fn sample_ncp_negative_is_nan() {
    let mut s = SimdNonCentralChiSquared::<f64>::new(3.0).seeded(&Unseeded);
    assert!(s.sample_ncp(-1.0).is_nan());
  }

  /// Non-negative `ncp` (including exactly zero) must stay finite.
  #[test]
  fn sample_ncp_nonnegative_is_finite() {
    let mut s = SimdNonCentralChiSquared::<f64>::new(3.0).seeded(&Unseeded);
    assert!(s.sample_ncp(0.0).is_finite());
    assert!(s.sample_ncp(2.5).is_finite());
  }

  fn assert_cumulant_moments(samples: &[f64], df: f64, lambda: f64) {
    let n = samples.len();
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
      "df = {df}: mean {mean} vs expected {expected_mean} (6*SE = {})",
      6.0 * se_mean
    );
    assert!(
      (var - expected_var).abs() < 6.0 * se_var,
      "df = {df}: variance {var} vs expected {expected_var} (6*SE = {})",
      6.0 * se_var
    );
  }

  /// `0 < df < 1` takes the Poisson mixture, mean `df + λ` and variance `2(df + 2λ)`; treating `df` as one would
  /// give 3.0 / 10.0 here against 2.3 / 8.6, tens of standard errors off.
  #[test]
  fn sample_ncp_df_below_one_matches_closed_form_moments() {
    let mut dist = SimdNonCentralChiSquared::<f64>::new(0.3).seeded(&Deterministic::new(11));
    let samples = (0..100_000)
      .map(|_| dist.sample_ncp(2.0))
      .collect::<Vec<_>>();
    assert_cumulant_moments(&samples, 0.3, 2.0);
  }

  /// The honest draw on the caller's rng has the same cumulant moments on both the shift and the mixture path.
  #[test]
  fn sample_ncp_with_matches_closed_form_moments() {
    for (df, lambda) in [(3.0, 2.5), (0.3, 2.0)] {
      let d = SimdNonCentralChiSquared::<f64>::new(df);
      let mut rng = SimdRng::from_seed(2718);
      let samples = (0..100_000)
        .map(|_| d.sample_ncp_with(&mut rng, lambda))
        .collect::<Vec<_>>();
      assert_cumulant_moments(&samples, df, lambda);
    }
  }
}
