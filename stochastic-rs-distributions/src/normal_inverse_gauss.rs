//! # Normal Inverse Gauss
//!
//! $$
//! X\sim\mathrm{Nig}(\alpha,\beta,\delta,\mu),\ \psi(u)=\mu u+\delta\left(\sqrt{\alpha^2-\beta^2}-\sqrt{\alpha^2-(\beta+iu)^2}\right)
//! $$
//!
//! Sampling: `μ + βW + √W·Z` with `W ~ IG(δ/γ, δ²)`, Barndorff-Nielsen, O.E. (1997), "Normal Inverse Gaussian Distributions and Stochastic Volatility Modelling", *Scandinavian Journal of Statistics* 24(1), 1-13, DOI 10.1111/1467-9469.00045.

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::SimdFloatExt;
use super::inverse_gauss::NormalPlusOwnState;
use super::inverse_gauss::SimdInverseGauss;
use super::normal::SimdNormal;
use crate::seeded::Buffered;
use crate::seeded::StreamState;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_NIG_THRESHOLD: usize = 16;

/// Normal-inverse-Gaussian law `NIG(alpha, beta, delta, mu)`: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdNormalInverseGauss<T> {
  alpha: T,
  beta: T,
  delta: T,
  mu: T,
  ig: SimdInverseGauss<T>,
}

/// A NIG stream: the inverse-Gaussian and normal sub-streams and the single-draw buffer.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct NigState<T: SimdFloatExt, R: SimdRngExt> {
  ig: NormalPlusOwnState<T, R>,
  normal: StreamState<T, R, 64>,
  buf: Buffered<T, 16>,
}

impl<T: SimdFloatExt> SimdNormalInverseGauss<T> {
  /// Creates a normal-inverse-Gaussian distribution via the Gaussian
  /// mixture `X = mu + beta·d + sqrt(d)·Z`, `d` drawn from an internal
  /// inverse-Gaussian subordinator.
  ///
  /// - `alpha` — tail heaviness α > 0, must exceed `|beta|` (matches the
  ///   module header's α).
  /// - `beta` — asymmetry β (matches the module header's β; a distinct
  ///   role from the `beta` shape parameter in
  ///   [`SimdBeta`](crate::beta::SimdBeta)/[`SimdGamma`](crate::gamma::SimdGamma)
  ///   or the skewness in [`crate::alpha_stable::SimdAlphaStable`]).
  /// - `delta` — scale δ > 0 (matches the module header's δ). Note this
  ///   does **not** pass straight through as the subordinator's own
  ///   mean/shape: internally `gamma = sqrt(alpha²−beta²)`, and the
  ///   inverse-Gaussian subordinator is built with mean `delta/gamma`
  ///   and shape `delta²`.
  /// - `mu` — location/drift μ (matches the module header's μ);
  ///   `mean() = mu + delta·beta/gamma`.
  pub fn new(alpha: T, beta: T, delta: T, mu: T) -> Self {
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(
      beta.is_finite(),
      "beta must satisfy `beta.is_finite()`, got beta = {beta:?}"
    );
    assert!(
      delta.is_finite(),
      "delta must satisfy `delta.is_finite()`, got delta = {delta:?}"
    );
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      alpha > beta.abs(),
      "alpha must satisfy `alpha > beta.abs()`, got alpha = {alpha:?}, beta = {beta:?}"
    );
    assert!(
      delta > T::zero(),
      "delta must satisfy `delta > T::zero()`, got delta = {delta:?}"
    );
    let ig_mu = delta / (alpha * alpha - beta * beta).sqrt();
    let ig_lambda = delta * delta;
    assert!(
      ig_mu > T::zero() && ig_mu.is_finite(),
      "delta must satisfy `0 < delta / (alpha * alpha - beta * beta).sqrt() < ∞`, got delta = {delta:?}, alpha = {alpha:?}, beta = {beta:?}"
    );
    assert!(
      ig_lambda > T::zero() && ig_lambda.is_finite(),
      "delta must satisfy `0 < delta * delta < ∞`, got delta = {delta:?}"
    );
    Self {
      alpha,
      beta,
      delta,
      mu,
      ig: SimdInverseGauss::new(ig_mu, ig_lambda),
    }
  }

  /// The tail heaviness `α`.
  pub fn alpha(&self) -> T {
    self.alpha
  }

  /// The asymmetry `β`.
  pub fn beta(&self) -> T {
    self.beta
  }

  /// The scale `δ`.
  pub fn delta(&self) -> T {
    self.delta
  }

  /// The location `μ`.
  pub fn mu(&self) -> T {
    self.mu
  }

  fn fill_parts<R: SimdRngExt>(
    &self,
    ig: &mut NormalPlusOwnState<T, R>,
    normal: &mut StreamState<T, R, 64>,
    out: &mut [T],
  ) {
    if out.len() < SMALL_NIG_THRESHOLD {
      for x in out.iter_mut() {
        let d = self.ig.next(ig);
        let z = SimdNormal::<T>::standard().next(normal);
        *x = self.mu + self.beta * d + d.sqrt() * z;
      }
      return;
    }
    let mu = T::splat(self.mu);
    let beta = T::splat(self.beta);
    let mut dbuf = [T::zero(); 64];
    let mut zbuf = [T::zero(); 64];
    let (chunks, rem) = out.as_chunks_mut::<64>();
    for chunk in chunks {
      self.ig.fill(ig, &mut dbuf);
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut zbuf);
      for (sub, (d8, z8)) in chunk.as_chunks_mut::<8>().0.iter_mut().zip(
        dbuf
          .as_chunks::<8>()
          .0
          .iter()
          .zip(zbuf.as_chunks::<8>().0.iter()),
      ) {
        let d = T::simd_from_array(*d8);
        let z = T::simd_from_array(*z8);
        let x = mu + beta * d + T::simd_sqrt(d) * z;
        *sub = T::simd_to_array(x);
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      self.ig.fill(ig, &mut dbuf[..n]);
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut zbuf[..n]);
      for i in 0..n {
        let d = dbuf[i];
        let z = zbuf[i];
        rem[i] = self.mu + self.beta * d + d.sqrt() * z;
      }
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    let d = self.ig.draw_with(rng);
    let z = SimdNormal::<T>::standard().draw_with(rng);
    self.mu + self.beta * d + d.sqrt() * z
  }
}

impl<T: SimdFloatExt> Sealed for SimdNormalInverseGauss<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdNormalInverseGauss<T> {
  type State<R: SimdRngExt> = NigState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (NigState<T, R>, u64) {
    let (ig, _) = self.ig.init::<R, S>(seed);
    let (normal, _) = SimdNormal::<T>::standard().init::<R, S>(seed);
    let basis = seed.next_seed();
    (
      NigState {
        ig,
        normal,
        buf: Buffered::new(),
      },
      basis,
    )
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdNormalInverseGauss<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut NigState<T, R>, out: &mut [T]) {
    self.fill_parts(&mut state.ig, &mut state.normal, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut NigState<T, R>) -> T {
    let NigState { ig, normal, buf } = state;
    buf.pop(|b| self.fill_parts(ig, normal, b))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdNormalInverseGauss<T> {
  /// `μ + βW + √W·Z` from one scalar inverse-Gaussian and one scalar normal draw on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdNormalInverseGauss<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    // f(x) = (αδ/π) exp(δγ + β(x−μ)) K₁(α q(x)) / q(x), γ = sqrt(α²−β²),
    // q(x) = sqrt(δ² + (x−μ)²). Barndorff-Nielsen (1997) eq. 3.
    //
    // Evaluated via K₁ᵉ(z) = e^z K₁(z) as
    // exp(δγ + β(x−μ) − α q(x)) · K₁ᵉ(α q(x)) rather than
    // exp(δγ + β(x−μ)) · K₁(α q(x)): for large |x−μ| the naive exponent
    // overflows while K₁'s far branch underflows, and ∞ · 0 = NaN. The
    // combined exponent is bounded above by δγ for every x, since
    // α q(x) ≥ α|x−μ| ≥ |β(x−μ)| (q(x) ≥ |x−μ|, α > |β| by construction),
    // so it underflows to 0 gracefully instead.
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let m = self.mu.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    let q = (d * d + (x - m) * (x - m)).sqrt();
    Some(
      a * d / std::f64::consts::PI
        * (d * gamma + b * (x - m) - a * q).exp()
        * crate::special::bessel_k1e(a * q)
        / q,
    )
  }

  fn mean(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let m = self.mu.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    Some(m + d * b / gamma)
  }

  fn variance(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    Some(d * a * a / gamma.powi(3))
  }

  fn skewness(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    Some(3.0 * b / (a * (d * gamma).sqrt()))
  }

  fn kurtosis(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    Some(3.0 * (1.0 + 4.0 * b * b / (a * a)) / (d * gamma))
  }

  fn characteristic_function(&self, t: f64) -> Option<num_complex::Complex64> {
    // φ(t) = exp{ iμt + δ (γ - sqrt(α² - (β + it)²)) },  γ = sqrt(α² - β²)
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let m = self.mu.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    let beta_plus_it = num_complex::Complex64::new(b, t);
    let inner = num_complex::Complex64::new(a * a, 0.0) - beta_plus_it * beta_plus_it;
    let exponent = num_complex::Complex64::new(0.0, m * t)
      + (num_complex::Complex64::new(gamma, 0.0) - inner.sqrt()).scale(d);
    Some(exponent.exp())
  }

  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    // M(t) = exp{ μt + δ (γ - sqrt(α² - (β + t)²)) }
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let m = self.mu.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    let bt = b + t;
    let inner = a * a - bt * bt;
    if inner < 0.0 {
      Some(f64::INFINITY)
    } else {
      Some((m * t + d * (gamma - inner.sqrt())).exp())
    }
  }
}

py_distribution!(PyNormalInverseGauss, SimdNormalInverseGauss,
  sig: (alpha, beta, delta, mu, seed=None, dtype=None),
  params: (alpha: f64, beta: f64, delta: f64, mu: f64)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::SimdNormalInverseGauss;
  use crate::tests::assert_ecf_matches;
  use crate::tests::assert_moments_within;
  use crate::tests::scalar_draws;
  use crate::traits::DistributionExt;
  use crate::traits::DistributionSampler;
  use crate::traits::SimdDistribution;

  #[test]
  fn scalar_sample_matches_moments_and_characteristic_function() {
    let d = SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.1);
    let xs = scalar_draws(&d, 17, 200_000);
    assert_moments_within(
      &xs,
      d.mean().unwrap(),
      d.variance().unwrap(),
      None,
      6.0,
      "NIG",
    );
    assert_ecf_matches(&xs, |u| d.characteristic_function(u).unwrap(), "NIG");
  }

  fn trapezoid(lo: f64, hi: f64, n: usize, mut f: impl FnMut(f64) -> f64) -> f64 {
    let h = (hi - lo) / n as f64;
    let mut integral = 0.0;
    let mut prev = f(lo);
    for i in 1..=n {
      let x = lo + h * i as f64;
      let cur = f(x);
      integral += 0.5 * (prev + cur) * h;
      prev = cur;
    }
    integral
  }

  /// pdf integrates to 1: trapezoid over μ±40δ grid, tol 1e-6 (α=2, β=0.5, δ=1, μ=0).
  #[test]
  fn nig_pdf_integrates_to_one() {
    let dist = SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.0);
    let integral = trapezoid(-40.0, 40.0, 400_000, |x| dist.pdf(x).unwrap());
    assert!(
      (integral - 1.0).abs() < 1e-6,
      "NIG pdf integral = {integral}, expected 1.0"
    );
  }

  /// First moment of pdf matches the existing closed-form mean() = μ + δβ/γ, tol 1e-5.
  #[test]
  fn nig_pdf_first_moment_matches_mean() {
    let dist = SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.0);
    let first_moment = trapezoid(-40.0, 40.0, 400_000, |x| x * dist.pdf(x).unwrap());
    let expected = dist.mean().unwrap();
    assert!(
      (first_moment - expected).abs() < 1e-5,
      "first moment = {first_moment}, mean() = {expected}"
    );
  }

  #[test]
  fn nig_pdf_symmetric_when_beta_zero() {
    let mu = 0.5;
    let dist = SimdNormalInverseGauss::<f64>::new(2.0, 0.0, 1.0, mu);
    for &h in &[0.1_f64, 0.5, 1.0, 2.0, 5.0] {
      let left = dist.pdf(mu - h).unwrap();
      let right = dist.pdf(mu + h).unwrap();
      assert!(
        (left - right).abs() < 1e-14,
        "pdf not symmetric at h={h}: f(mu-h)={left}, f(mu+h)={right}"
      );
    }
  }

  /// Regression: naively computing `exp(δγ+β(x−μ)) · K₁(α q(x))` overflows
  /// the `exp` term (exponent ≈1001.9 > 709.78) while `K₁`'s far branch
  /// underflows `exp(−4000)` to exact `0.0`, producing `∞ · 0 = NaN`. The
  /// cancellation-safe `K₁ᵉ` route must instead underflow cleanly to `0.0`
  /// (the true limiting density at this deviation is far below f64's
  /// smallest positive value).
  #[test]
  fn nig_pdf_finite_for_large_deviation() {
    let dist = SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.0);
    let p = dist.pdf(2000.0).unwrap();
    assert!(p.is_finite(), "pdf(2000.0) = {p}, expected a finite value");
    assert_eq!(
      p, 0.0,
      "pdf(2000.0) should underflow cleanly to 0.0, got {p}"
    );
  }

  /// The high-skew regime (β/α close to 1) moves the overflow trigger
  /// inward; sweep a range of moderately large deviations and require
  /// every value to be finite and non-negative (a valid density never
  /// goes negative or non-finite, however small).
  #[test]
  fn nig_pdf_finite_across_tail_sweep() {
    let dist = SimdNormalInverseGauss::<f64>::new(2.0, 1.9, 1.0, 0.0);
    for i in 1..=200 {
      let x = i as f64 * 10.0;
      let p = dist.pdf(x).unwrap();
      assert!(
        p.is_finite() && p >= 0.0,
        "pdf({x}) = {p}, expected finite >= 0"
      );
    }
  }

  /// A fill below `SMALL_NIG_THRESHOLD` pops both sub-streams' buffers instead of filling from their engines.
  #[test]
  fn nig_fill_slice_small_n_is_deterministic_and_finite() {
    let dist = SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.0);
    let mut out_a = [0.0_f64; 8];
    let mut out_b = [0.0_f64; 8];
    dist.seeded(&Deterministic::new(7)).fill_slice(&mut out_a);
    dist.seeded(&Deterministic::new(7)).fill_slice(&mut out_b);
    assert_eq!(out_a, out_b, "same seed must replay bit-for-bit");
    assert!(
      out_a.iter().all(|x| x.is_finite()),
      "all samples must be finite, got {out_a:?}"
    );
  }
}
