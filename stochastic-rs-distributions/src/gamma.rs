//! # Gamma
//!
//! $$
//! f(x)=\frac{1}{\theta^\alpha\Gamma(\alpha)}x^{\alpha-1}e^{-x/\theta},\ x>0
//! $$
//!
//! Scale parametrization (mean = αθ; NOT the rate form `β=1/θ` — the
//! `scale` constructor argument below is θ, not β).
//!
//! Sampling: Marsaglia-Tsang squeeze method over the buffered SIMD normal
//! and uniform sources. The squeeze loop itself stays scalar — an 8-lane
//! batched variant was measured slower on 128-bit SIMD targets, where the
//! lane bookkeeping outweighs the vectorised arithmetic and the inputs are
//! already SIMD-amortised. `α < 1` is boosted via
//! $\mathrm{Gamma}(\alpha) = \mathrm{Gamma}(\alpha+1) \cdot U^{1/\alpha}$.
//!
//! Reference: Marsaglia, G., Tsang, W.W. (2000), "A simple method for
//! generating gamma variables", *ACM TOMS* 26(3), 363-372,
//! DOI: 10.1145/358407.358414.
use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::SimdFloatExt;
use super::normal::SimdNormal;
use crate::seeded::Buffered;
use crate::seeded::StreamState;
use crate::source::AnyRng;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Gamma law `Gamma(alpha, scale)`: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdGamma<T> {
  alpha: T,
  scale: T,
}

/// A gamma stream: the normal sub-stream the squeeze pops, the uniform engine and the single-draw buffer.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct GammaState<T: SimdFloatExt, R: SimdRngExt> {
  normal: StreamState<T, R, 64>,
  rng: R,
  buf: Buffered<T, 16>,
}

/// The standard normal and the uniform a Marsaglia–Tsang trial consumes, from a stream or the caller's rng.
trait MtSource<T> {
  fn normal(&mut self) -> T;

  fn uniform(&mut self) -> T;
}

impl<T: SimdFloatExt, R: SimdRngExt> MtSource<T> for (&mut StreamState<T, R, 64>, &mut R) {
  #[inline]
  fn normal(&mut self) -> T {
    SimdNormal::<T>::standard().next(self.0)
  }

  #[inline]
  fn uniform(&mut self) -> T {
    T::sample_uniform_simd(self.1)
  }
}

impl<T: SimdFloatExt, G: Rng + ?Sized> MtSource<T> for AnyRng<'_, G> {
  #[inline]
  fn normal(&mut self) -> T {
    SimdNormal::<T>::standard().draw_with(self.0)
  }

  #[inline]
  fn uniform(&mut self) -> T {
    T::sample_uniform(self.0)
  }
}

impl<T: SimdFloatExt> SimdGamma<T> {
  /// Creates a gamma distribution.
  ///
  /// - `alpha` — shape α > 0 (matches the module header's α).
  /// - `scale` — scale θ > 0 (matches the module header's θ; mean =
  ///   α·θ). This is the scale parametrization — pass `1.0 / rate` if
  ///   you have a rate-parametrized β instead.
  pub fn new(alpha: T, scale: T) -> Self {
    assert!(
      alpha > T::zero() && scale > T::zero(),
      "alpha must satisfy `alpha > T::zero() && scale > T::zero()`, got alpha = {alpha:?}, scale = {scale:?}"
    );
    Self { alpha, scale }
  }

  /// The shape `α`.
  pub fn alpha(&self) -> T {
    self.alpha
  }

  /// The scale `θ`.
  pub fn scale(&self) -> T {
    self.scale
  }

  /// Marsaglia–Tsang's `(d, c)` for `α` (`α + 1` below one), and `1/α` when the `U^{1/α}` boost applies.
  #[inline]
  fn squeeze(&self) -> (T, T, Option<T>) {
    let third = T::from(1.0 / 3.0).unwrap();
    let nine = T::from(9.0).unwrap();
    let boosted = self.alpha < T::one();
    let alpha_eff = if boosted {
      self.alpha + T::one()
    } else {
      self.alpha
    };
    let d = alpha_eff - third;
    let c = T::one() / (nine * d).sqrt();
    (d, c, boosted.then(|| T::one() / self.alpha))
  }

  /// One unscaled Marsaglia–Tsang draw `d·v` of `Gamma(d + 1/3, 1)`.
  #[inline(always)]
  fn mt_one<S: MtSource<T>>(src: &mut S, d: T, c: T) -> T {
    let c1 = T::from(0.0331).unwrap();
    let half = T::from(0.5).unwrap();
    loop {
      let z = src.normal();
      let t = T::one() + c * z;
      let v = t * t * t;
      if v <= T::zero() {
        continue;
      }
      let u = src.uniform();
      let z2 = z * z;
      if u < T::one() - c1 * z2 * z2 {
        return d * v;
      }
      if u.ln() < half * z2 + d * (T::one() - v + v.ln()) {
        return d * v;
      }
    }
  }

  #[inline(always)]
  fn draw<S: MtSource<T>>(&self, src: &mut S, d: T, c: T, inv_alpha: Option<T>) -> T {
    let g = Self::mt_one(src, d, c);
    match inv_alpha {
      Some(inv_alpha) => self.scale * g * src.uniform().powf(inv_alpha),
      None => self.scale * g,
    }
  }

  /// Kept out of line because inlined into a pop loop's refill this rejection kernel slows the loop.
  #[inline(never)]
  fn fill_parts<R: SimdRngExt>(
    &self,
    normal: &mut StreamState<T, R, 64>,
    rng: &mut R,
    out: &mut [T],
  ) {
    let (d, c, inv_alpha) = self.squeeze();
    let mut src = (normal, rng);
    // One loop per branch, so the boost test is not paid per draw.
    match inv_alpha {
      Some(inv_alpha) => {
        for x in out.iter_mut() {
          *x = self.draw(&mut src, d, c, Some(inv_alpha));
        }
      }
      None => {
        for x in out.iter_mut() {
          *x = self.draw(&mut src, d, c, None);
        }
      }
    }
  }

  fn log_draw<S: MtSource<T>>(&self, src: &mut S) -> T {
    let (d, c, inv_alpha) = self.squeeze();
    let log_core = self.scale.ln() + Self::mt_one(src, d, c).ln();
    if inv_alpha.is_some() {
      // `1 - u` lands the uniform in `(0, 1]`, so the log is finite where
      // the generator's own `[0, 1)` would have handed back a `-inf`.
      let u = T::one() - src.uniform();
      log_core + u.ln() / self.alpha
    } else {
      log_core
    }
  }

  /// `ln` of one draw without the `u^{1/α}` factor, which underflows to zero at small `α`: the Beta and Dirichlet `0/0` repair.
  pub(crate) fn next_log<R: SimdRngExt>(&self, state: &mut GammaState<T, R>) -> T {
    self.log_draw(&mut (&mut state.normal, &mut state.rng))
  }

  pub(crate) fn draw_log_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.log_draw(&mut AnyRng(rng))
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    let (d, c, inv_alpha) = self.squeeze();
    self.draw(&mut AnyRng(rng), d, c, inv_alpha)
  }
}

/// Gamma(shape=2, scale=2) — mean 4, matching the repeated Gamma fixture in
/// the umbrella crate's workspace-root `benches/distributions.rs` and
/// `benches/dist_multicore.rs` (not this crate's own — `stochastic-rs-
/// distributions` has no `benches/` directory of its own).
impl<T: SimdFloatExt> Default for SimdGamma<T> {
  fn default() -> Self {
    Self::new(T::from(2.0).unwrap(), T::from(2.0).unwrap())
  }
}

impl<T: SimdFloatExt> Sealed for SimdGamma<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdGamma<T> {
  type State<R: SimdRngExt> = GammaState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (GammaState<T, R>, u64) {
    let (normal, _) = SimdNormal::<T>::standard().init::<R, S>(seed);
    let stream_seed = seed.next_seed();
    (
      GammaState {
        normal,
        rng: R::from_seed(stream_seed),
        buf: Buffered::new(),
      },
      stream_seed,
    )
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdGamma<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut GammaState<T, R>, out: &mut [T]) {
    self.fill_parts(&mut state.normal, &mut state.rng, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut GammaState<T, R>) -> T {
    let GammaState { normal, rng, buf } = state;
    buf.pop(|b| self.fill_parts(normal, rng, b))
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdGamma<T> {
  fn pdf(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    }
    let alpha = self.alpha.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    // f(x) = x^(α−1) e^(−x/θ) / (θ^α Γ(α))
    let log_pdf =
      (alpha - 1.0) * x.ln() - x / scale - alpha * scale.ln() - crate::special::ln_gamma(alpha);
    log_pdf.exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    }
    let alpha = self.alpha.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    crate::special::gamma_p(alpha, x / scale)
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    // Newton-bisection hybrid on the CDF.
    if p <= 0.0 {
      return 0.0;
    }
    if p >= 1.0 {
      return f64::INFINITY;
    }
    let alpha = self.alpha.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    // Start from the Wilson-Hilferty Gaussian approximation.
    let z = crate::special::ndtri(p);
    let mut x = alpha * (1.0 - 1.0 / (9.0 * alpha) + z / (3.0 * alpha.sqrt())).powi(3);
    if x <= 0.0 {
      x = 0.5 * alpha;
    }
    x *= scale;
    // 30 Newton iterations using f(x) = P(α, x/θ) − p, f'(x) = pdf(x).
    for _ in 0..30 {
      let f = crate::special::gamma_p(alpha, x / scale) - p;
      let pdf =
        ((alpha - 1.0) * x.ln() - x / scale - alpha * scale.ln() - crate::special::ln_gamma(alpha))
          .exp();
      if pdf <= 0.0 {
        break;
      }
      let dx = f / pdf;
      let new_x = (x - dx).max(x * 1e-12);
      if (new_x - x).abs() < 1e-14 * x.max(1.0) {
        return new_x;
      }
      x = new_x;
    }
    x
  }

  fn mean(&self) -> f64 {
    self.alpha.to_f64().unwrap() * self.scale.to_f64().unwrap()
  }

  fn mode(&self) -> f64 {
    let alpha = self.alpha.to_f64().unwrap();
    if alpha < 1.0 {
      0.0
    } else {
      (alpha - 1.0) * self.scale.to_f64().unwrap()
    }
  }

  fn variance(&self) -> f64 {
    let alpha = self.alpha.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    alpha * scale * scale
  }

  fn skewness(&self) -> f64 {
    let alpha = self.alpha.to_f64().unwrap();
    2.0 / alpha.sqrt()
  }

  fn kurtosis(&self) -> f64 {
    // Excess kurtosis.
    let alpha = self.alpha.to_f64().unwrap();
    6.0 / alpha
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    let alpha = self.alpha.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    if t < 1.0 / scale {
      (1.0 - scale * t).powf(-alpha)
    } else {
      f64::INFINITY
    }
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    // φ(t) = (1 − i θ t)^{−α}
    let alpha = self.alpha.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    let denom = num_complex::Complex64::new(1.0, -scale * t);
    denom.powf(-alpha)
  }

  fn entropy(&self) -> f64 {
    let alpha = self.alpha.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    alpha
      + scale.ln()
      + crate::special::ln_gamma(alpha)
      + (1.0 - alpha) * crate::special::digamma(alpha)
  }

  fn median(&self) -> f64 {
    self.inv_cdf(0.5)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdGamma<T> {
  /// One scalar Marsaglia–Tsang draw on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

py_distribution!(PyGamma, SimdGamma,
  sig: (alpha, scale, seed=None, dtype=None),
  params: (alpha: f64, scale: f64)
);

#[cfg(test)]
mod tests {
  use ndarray::ArrayView1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::KolmogorovSmirnovConfig;
  use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::kolmogorov_smirnov_test;

  use super::SimdGamma;
  use crate::tests::scalar_ks_best_p;
  use crate::traits::DistributionExt as _;
  use crate::traits::DistributionSampler;
  use crate::traits::SimdDistribution;

  /// Both tests below check KS against the sampler's own `cdf`
  /// (Kolmogorov 1933 / Smirnov 1948 / Massey 1951 critical values,
  /// alpha=0.05 — see
  /// `stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov`'s module
  /// doc), best-of-three pinned seeds: a correct test still rejects a
  /// true null at rate alpha, and the SIMD stream differs across
  /// platforms, so one seed cannot be trusted to be lucky everywhere.
  /// Replaces this test's own former `ks_critical = 2.0/sqrt(N)` bound,
  /// which implied an undeclared alpha of roughly 0.0007.
  #[test]
  fn simd_gamma_fill_matches_theoretical_distribution() {
    const N: usize = 40_000;
    let best_p = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let dist = SimdGamma::<f64>::new(2.5, 1.5);
        let mut samples = vec![0.0_f64; N];
        dist
          .seeded(&Deterministic::new(seed))
          .fill_slice(&mut samples);
        assert!(samples.iter().all(|x| x.is_finite() && *x > 0.0));
        kolmogorov_smirnov_test(
          ArrayView1::from(&samples),
          |x| dist.cdf(x),
          KolmogorovSmirnovConfig::default(),
        )
        .p_value
      })
      .fold(0.0_f64, f64::max);
    assert!(
      best_p > 0.01,
      "every seed gave p <= 0.01 (best {best_p}); likely a bug, not bad luck"
    );
  }

  #[test]
  fn simd_gamma_boosted_alpha_below_one_matches_theory() {
    const N: usize = 40_000;
    let best_p = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let dist = SimdGamma::<f64>::new(0.5, 2.0);
        let mut samples = vec![0.0_f64; N];
        dist
          .seeded(&Deterministic::new(seed))
          .fill_slice(&mut samples);
        assert!(samples.iter().all(|x| x.is_finite() && *x >= 0.0));
        kolmogorov_smirnov_test(
          ArrayView1::from(&samples),
          |x| dist.cdf(x),
          KolmogorovSmirnovConfig::default(),
        )
        .p_value
      })
      .fold(0.0_f64, f64::max);
    assert!(
      best_p > 0.01,
      "every seed gave p <= 0.01 (best {best_p}); likely a bug, not bad luck"
    );
  }

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf, boosted shape included.
  #[test]
  fn scalar_sample_matches_cdf() {
    for (alpha, scale) in [(2.5, 1.5), (0.5, 2.0)] {
      let d = SimdGamma::<f64>::new(alpha, scale);
      let best = scalar_ks_best_p(&d, |x| d.cdf(x));
      assert!(best > 0.01, "Gamma({alpha}, {scale}): best p = {best}");
    }
  }
}
