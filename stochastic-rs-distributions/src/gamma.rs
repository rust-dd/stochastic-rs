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
use std::cell::Cell;
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use super::normal::SimdNormal;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;

pub struct SimdGamma<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  alpha: T,
  scale: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  normal: SimdNormal<T, 64, R>,
  simd_rng: UnsafeCell<R>,
  pub(crate) stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdGamma<T, R> {
  /// Creates a gamma distribution.
  ///
  /// - `alpha` — shape α > 0 (matches the module header's α).
  /// - `scale` — scale θ > 0 (matches the module header's θ; mean =
  ///   α·θ). This is the scale parametrization — pass `1.0 / rate` if
  ///   you have a rate-parametrized β instead.
  ///
  /// RNGs come from a [`SeedExt`](crate::simd_rng::SeedExt) source; each
  /// sub-component (normal, main rng) gets an independent stream.
  pub fn new<S: crate::simd_rng::SeedExt>(alpha: T, scale: T, seed: &S) -> Self {
    assert!(
      alpha > T::zero() && scale > T::zero(),
      "alpha must satisfy `alpha > T::zero() && scale > T::zero()`, got alpha = {alpha:?}, scale = {scale:?}"
    );
    let normal = SimdNormal::<T, 64, R>::new(T::zero(), T::one(), seed);
    let stream_seed = seed.seed_value();
    Self {
      alpha,
      scale,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      normal,
      simd_rng: UnsafeCell::new(R::from_seed(stream_seed)),
      stream_seed: Cell::new(stream_seed),
    }
  }

  /// Builds an independent worker stream for the `stream_idx`-th chunk of a
  /// parallel `sample_matrix` fan-out; see
  /// [`DistributionSampler::fork`](crate::traits::DistributionSampler::fork).
  #[doc(hidden)]
  pub fn fork(&self, stream_idx: u64) -> Self {
    let mut basis = self.stream_seed.get();
    let call_basis = crate::simd_rng::derive_seed(&mut basis);
    self.stream_seed.set(basis);
    let child_seed = crate::simd_rng::derive_fork_seed(call_basis, stream_idx);
    Self::new(
      self.alpha,
      self.scale,
      &crate::simd_rng::Deterministic::new(child_seed),
    )
  }

  /// Returns a single sample using the internal SIMD RNG.
  #[inline]
  pub fn sample_fast(&self) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 16 {
      self.refill_buffer();
    }
    let buf = unsafe { &mut *self.buffer.get() };
    let z = buf[*index];
    *index += 1;
    z
  }

  /// One scalar Marsaglia-Tsang draw of `d·v` (unscaled `Gamma(α_eff, 1)`),
  /// used by short fills, tails and the rare SIMD lane rejections.
  #[inline]
  fn sample_mt_one(rng: &mut R, normal: &SimdNormal<T, 64, R>, d: T, c: T) -> T {
    let c1 = T::from(0.0331).unwrap();
    let half = T::from(0.5).unwrap();
    loop {
      let z = normal.sample_fast();
      let t = T::one() + c * z;
      let v = t * t * t;
      if v <= T::zero() {
        continue;
      }
      let u = T::sample_uniform_simd(rng);
      let z2 = z * z;
      if u < T::one() - c1 * z2 * z2 {
        return d * v;
      }
      if u.ln() < half * z2 + d * (T::one() - v + v.ln()) {
        return d * v;
      }
    }
  }

  /// Fills `out` using the internal SIMD RNG stream — the only stream this
  /// sampler draws from (see the crate-level RNG policy).
  pub fn fill_slice(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
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

    if boosted {
      let inv_alpha = T::one() / self.alpha;
      for x in out.iter_mut() {
        let g = Self::sample_mt_one(rng, &self.normal, d, c);
        let u = T::sample_uniform_simd(rng);
        *x = self.scale * g * u.powf(inv_alpha);
      }
    } else {
      for x in out.iter_mut() {
        *x = self.scale * Self::sample_mt_one(rng, &self.normal, d, c);
      }
    }
  }

  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice(buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

/// Gamma(shape=2, scale=2) — mean 4, matching the repeated Gamma fixture in
/// the umbrella crate's workspace-root `benches/distributions.rs` and
/// `benches/dist_multicore.rs` (not this crate's own — `stochastic-rs-
/// distributions` has no `benches/` directory of its own).
impl<T: SimdFloatExt, R: SimdRngExt> Default for SimdGamma<T, R> {
  fn default() -> Self {
    Self::new(T::from(2.0).unwrap(), T::from(2.0).unwrap(), &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdGamma<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.alpha, self.scale, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdGamma<T, R> {
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

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdGamma<T, R> {
  /// The `rng` argument is intentionally unused — this type draws from its
  /// own internal SIMD stream seeded at construction. Use `Deterministic`
  /// in the constructor for reproducibility.
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
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
  use crate::traits::DistributionExt as _;

  /// Both tests below check KS against the sampler's own `cdf`
  /// (Kolmogorov 1933 / Smirnov 1948 / Massey 1951 critical values,
  /// alpha=0.05 — see
  /// `stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov`'s module
  /// doc), worst-of-three pinned seeds: a correct test still rejects a
  /// true null at rate alpha, and the SIMD stream differs across
  /// platforms, so one seed cannot be trusted to be lucky everywhere.
  /// Replaces this test's own former `ks_critical = 2.0/sqrt(N)` bound,
  /// which implied an undeclared alpha of roughly 0.0007.
  #[test]
  fn simd_gamma_fill_matches_theoretical_distribution() {
    const N: usize = 40_000;
    let worst_p = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let dist = SimdGamma::<f64>::new(2.5, 1.5, &Deterministic::new(seed));
        let mut samples = vec![0.0_f64; N];
        dist.fill_slice(&mut samples);
        assert!(samples.iter().all(|x| x.is_finite() && *x > 0.0));
        kolmogorov_smirnov_test(
          ArrayView1::from(&samples),
          |x| dist.cdf(x),
          KolmogorovSmirnovConfig::default(),
        )
        .p_value
      })
      .fold(1.0_f64, f64::min);
    assert!(
      worst_p > 0.01,
      "every seed gave p <= 0.01 (worst {worst_p}); likely a bug, not bad luck"
    );
  }

  #[test]
  fn simd_gamma_boosted_alpha_below_one_matches_theory() {
    const N: usize = 40_000;
    let worst_p = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let dist = SimdGamma::<f64>::new(0.5, 2.0, &Deterministic::new(seed));
        let mut samples = vec![0.0_f64; N];
        dist.fill_slice(&mut samples);
        assert!(samples.iter().all(|x| x.is_finite() && *x >= 0.0));
        kolmogorov_smirnov_test(
          ArrayView1::from(&samples),
          |x| dist.cdf(x),
          KolmogorovSmirnovConfig::default(),
        )
        .p_value
      })
      .fold(1.0_f64, f64::min);
    assert!(
      worst_p > 0.01,
      "every seed gave p <= 0.01 (worst {worst_p}); likely a bug, not bad luck"
    );
  }
}
