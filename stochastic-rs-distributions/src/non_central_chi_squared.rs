//! # Non Central Chi Squared
//!
//! $$
//! X\sim\chi^2_\nu(\lambda),\quad f_X(x)=\tfrac12 e^{-(x+\lambda)/2}(x/\lambda)^{\nu/4-1/2}I_{\nu/2-1}(\sqrt{\lambda x})
//! $$
//!
//! Sampling: $\chi^2_{\nu-1} + (Z + \sqrt{\lambda})^2$ for $\nu \ge 1$, else $\mathrm{Gamma}(\nu/2 + J, 2)$, $J \sim \mathrm{Poisson}(\lambda/2)$, whose $J = 0$ term at $\nu = 0$ is the atom at 0.
//!
//! Reference: Johnson, N.L., Kotz, S., Balakrishnan, N. (1995), *Continuous Univariate Distributions*, vol. 2, 2nd ed., Wiley, ch. 29.
//! Reference: Abramowitz, M., Stegun, I.A. (1964), *Handbook of Mathematical Functions*, NBS AMS 55, eq. 26.4.25 (the Poisson mixture).
//! Reference: Siegel, A. F. (1979), "The noncentral chi-squared distribution with zero degrees of freedom and testing for uniformity", *Biometrika* 66(2), 381-386, DOI 10.1093/biomet/66.2.381.
use rand::Rng;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::SimdRngExt;
use stochastic_rs_core::simd_rng::derive_seed;

use crate::chi_square::SimdChiSquared;
use crate::gamma::GammaState;
use crate::gamma::SimdGamma;
use crate::normal::SimdNormal;
use crate::poisson::count_once;
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
  fn poisson(&mut self, lambda: f64) -> f64;

  fn gamma(&mut self, law: SimdGamma<T>) -> T;
}

impl<T: SimdFloatExt, S: SeedExt> MixtureSource<T> for &S {
  /// The first draw of a `SimdPoisson` stream on this seed, without the table and the buffer that stream would fill.
  fn poisson(&mut self, lambda: f64) -> f64 {
    count_once(lambda, &mut SimdRng::from_seed(self.next_seed()))
  }

  fn gamma(&mut self, law: SimdGamma<T>) -> T {
    law.seeded(*self).sample()
  }
}

impl<T: SimdFloatExt, G: Rng + ?Sized> MixtureSource<T> for AnyRng<'_, G> {
  fn poisson(&mut self, lambda: f64) -> f64 {
    count_once(lambda, self.0)
  }

  fn gamma(&mut self, law: SimdGamma<T>) -> T {
    law.draw_with(self.0)
  }
}

/// Panics unless `0 ≤ ncp < ∞`, so a negative, infinite or `NaN` noncentrality fails on both decompositions alike.
#[inline]
fn check_ncp<T: SimdFloatExt>(ncp: T) {
  assert!(
    ncp >= T::zero() && ncp.is_finite(),
    "ncp must satisfy `ncp >= T::zero() && ncp.is_finite()`, got ncp = {ncp:?}"
  );
}

/// The Poisson mixture `Gamma(df/2 + J, 2)`, `J ~ Poisson(ncp/2)`, exact for `0 ≤ df < 1`; `df = 0` with `J = 0` is Siegel's atom at 0.
fn poisson_mixture<T: SimdFloatExt>(df: T, ncp: T, mut src: impl MixtureSource<T>) -> T {
  let two = T::from_f64_fast(2.0);
  let half_lambda = (ncp / two).to_f64().unwrap_or(f64::NAN);
  let mixture_jumps = if half_lambda > 0.0 {
    src.poisson(half_lambda)
  } else {
    0.0
  };
  let shape = df / two + T::from_f64_fast(mixture_jumps);
  if shape == T::zero() {
    return T::zero();
  }
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
  /// The law of `χ²_df(·)`, whose central `χ²_{df−1}` term is dropped when `df ≈ 1`; panics unless `0 ≤ df < ∞`.
  pub fn new(df: T) -> Self {
    assert!(
      df.is_finite(),
      "df must satisfy `df.is_finite()`, got df = {df:?}"
    );
    assert!(
      df >= T::zero(),
      "df must satisfy `df >= T::zero()`, got df = {df:?}"
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

  /// One draw of `χ²_df(ncp)` on the caller's rng, the shift for `df ≥ 1` and the Poisson mixture below; panics
  /// unless `0 ≤ ncp < ∞`.
  pub fn sample_ncp_with<G: Rng + ?Sized>(&self, rng: &mut G, ncp: T) -> T {
    check_ncp(ncp);
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
  /// One draw of `χ²_df(ncp)` from the stream; panics unless `0 ≤ ncp < ∞`.
  #[inline]
  pub fn sample_ncp(&mut self, ncp: T) -> T {
    check_ncp(ncp);
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

/// One-shot `χ²_df(ncp)` draw, the shift for `df ≥ 1` and the Poisson mixture below; panics on the `df` that `new`
/// rejects or unless `0 ≤ ncp < ∞`. It builds the laws per call, so repeated draws belong on a stream.
pub fn sample<T: FloatExt, S: SeedExt>(df: T, ncp: T, seed: &S) -> T {
  let law = SimdNonCentralChiSquared::<T>::new(df);
  if df >= T::one() {
    return law.seeded(seed).sample_ncp(ncp);
  }
  check_ncp(ncp);
  poisson_mixture(df, ncp, seed)
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::SimdRng;

  use super::*;

  const SEEDS: [u64; 3] = [2718, 999, 42];

  /// A negative, infinite or `NaN` noncentrality panics naming `ncp`, on both decompositions and every entry point.
  #[test]
  fn negative_or_non_finite_ncp_is_rejected() {
    let names_ncp = |r: std::thread::Result<f64>| {
      r.err()
        .and_then(|e| e.downcast::<String>().ok())
        .is_some_and(|m| m.starts_with("ncp must satisfy"))
    };
    for seed in SEEDS {
      for df in [3.0, 0.3] {
        for ncp in [-1.0, f64::NAN, f64::INFINITY] {
          let law = SimdNonCentralChiSquared::<f64>::new(df);
          let seeded =
            std::panic::catch_unwind(|| law.seeded(&Deterministic::new(seed)).sample_ncp(ncp));
          let honest =
            std::panic::catch_unwind(|| law.sample_ncp_with(&mut SimdRng::from_seed(seed), ncp));
          let one_shot = std::panic::catch_unwind(|| sample(df, ncp, &Deterministic::new(seed)));
          assert!(
            names_ncp(seeded) && names_ncp(honest) && names_ncp(one_shot),
            "df = {df}, ncp = {ncp} was not rejected as `ncp`"
          );
        }
      }
    }
  }

  /// Siegel (1979): `χ²₀(λ)` is `Gamma(J, 2)` for `J ~ Poisson(λ/2)`, with the atom `P(X = 0) = e^{−λ/2}`.
  #[test]
  fn zero_degrees_of_freedom_is_siegels_law() {
    let mut s = SimdNonCentralChiSquared::<f64>::new(0.0).seeded(&Deterministic::new(11));
    let samples = (0..100_000).map(|_| s.sample_ncp(2.0)).collect::<Vec<_>>();
    let atom = samples.iter().filter(|&&x| x == 0.0).count() as f64 / samples.len() as f64;
    let want = (-1.0_f64).exp();
    assert!(
      (atom - want).abs() < 6.0 * (want * (1.0 - want) / 100_000.0).sqrt(),
      "atom {atom} vs {want}"
    );
    assert_cumulant_moments(&samples, 0.0, 2.0);
    assert!(samples.iter().all(|&x| x >= 0.0));
    let law = SimdNonCentralChiSquared::<f64>::new(0.0);
    let mut rng = SimdRng::from_seed(7);
    for _ in 0..100_000 {
      let x = law.sample_ncp_with(&mut rng, 2.0);
      assert!(x.is_finite() && x >= 0.0, "{x}");
    }
    assert_eq!(sample(0.0_f64, 0.0, &Deterministic::new(1)), 0.0);
    assert!(sample(0.0_f32, 2.0, &Deterministic::new(1)) >= 0.0);
  }

  /// A negative `df` panics naming `df >= T::zero()`; `df = 0` constructs.
  #[test]
  fn negative_df_is_rejected_zero_is_not() {
    let r = std::panic::catch_unwind(|| SimdNonCentralChiSquared::<f64>::new(-0.5));
    let msg = r.err().and_then(|e| e.downcast::<String>().ok()).unwrap();
    assert!(
      msg.starts_with("df must satisfy `df >= T::zero()`"),
      "{msg}"
    );
    assert_eq!(SimdNonCentralChiSquared::<f64>::new(0.0).df(), 0.0);
  }

  /// A noncentrality of `1e300` below `df = 1` draws its count from a few constants and as an `f64`, on every entry point.
  #[test]
  fn huge_noncentrality_terminates() {
    let law = SimdNonCentralChiSquared::<f64>::new(0.3);
    let seeded = law.seeded(&Deterministic::new(1)).sample_ncp(1e300);
    let honest = law.sample_ncp_with(&mut SimdRng::from_seed(1), 1e300);
    let one_shot = sample(0.3, 1e300, &Deterministic::new(1));
    for x in [seeded, honest, one_shot] {
      assert!(x.is_finite() && x > 1e299, "{x}");
    }
  }

  /// Non-negative `ncp` (including exactly zero) must stay finite.
  #[test]
  fn sample_ncp_nonnegative_is_finite() {
    for seed in SEEDS {
      let mut s = SimdNonCentralChiSquared::<f64>::new(3.0).seeded(&Deterministic::new(seed));
      assert!(s.sample_ncp(0.0).is_finite());
      assert!(s.sample_ncp(2.5).is_finite());
    }
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
    for (df, lambda) in [(3.0, 2.5), (0.3, 2.0), (0.0, 2.0)] {
      let d = SimdNonCentralChiSquared::<f64>::new(df);
      let mut rng = SimdRng::from_seed(2718);
      let samples = (0..100_000)
        .map(|_| d.sample_ncp_with(&mut rng, lambda))
        .collect::<Vec<_>>();
      assert_cumulant_moments(&samples, df, lambda);
    }
  }
}
