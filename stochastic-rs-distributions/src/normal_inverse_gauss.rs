//! # Normal Inverse Gauss
//!
//! $$
//! X\sim\mathrm{Nig}(\alpha,\beta,\delta,\mu),\ \psi(u)=\mu u+\delta\left(\sqrt{\alpha^2-\beta^2}-\sqrt{\alpha^2-(\beta+iu)^2}\right)
//! $$
//!
use std::cell::Cell;
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use super::inverse_gauss::SimdInverseGauss;
use super::normal::SimdNormal;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;

const SMALL_NIG_THRESHOLD: usize = 16;

pub struct SimdNormalInverseGauss<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  alpha: T,
  beta: T,
  delta: T,
  mu: T,
  ig: SimdInverseGauss<T, R>,
  normal: SimdNormal<T, 64, R>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdNormalInverseGauss<T, R> {
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
  pub fn new<S: crate::simd_rng::SeedExt>(alpha: T, beta: T, delta: T, mu: T, seed: &S) -> Self {
    assert!(
      alpha > T::zero() && alpha > beta.abs(),
      "Nig: alpha must be > |beta|"
    );
    assert!(delta > T::zero(), "Nig: delta must be positive");
    let gamma = (alpha * alpha - beta * beta).sqrt();
    let ig_mean = delta / gamma;
    let ig_shape = delta * delta;
    let ig = SimdInverseGauss::<T, R>::new(ig_mean, ig_shape, seed);
    let normal = SimdNormal::<T, 64, R>::new(T::zero(), T::one(), seed);
    let stream_seed = seed.seed_value();
    Self {
      alpha,
      beta,
      delta,
      mu,
      ig,
      normal,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
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
      self.beta,
      self.delta,
      self.mu,
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

  /// Fills `out` using the internal SIMD RNG stream — the only stream this
  /// sampler draws from (see the crate-level RNG policy). `ig` and
  /// `normal` each own and refill their own internal buffer, so this type
  /// needs no separate engine of its own (matches `SimdLogNormal`, which
  /// dropped its equivalent field for the same reason).
  pub fn fill_slice(&self, out: &mut [T]) {
    if out.len() < SMALL_NIG_THRESHOLD {
      for x in out.iter_mut() {
        let d = self.ig.sample_fast();
        let z = self.normal.sample_fast();
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
      self.ig.fill_slice(&mut dbuf);
      self.normal.fill_standard_fast(&mut zbuf);
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
      self.ig.fill_slice(&mut dbuf[..n]);
      self.normal.fill_standard_fast(&mut zbuf[..n]);
      for i in 0..n {
        let d = dbuf[i];
        let z = zbuf[i];
        rem[i] = self.mu + self.beta * d + d.sqrt() * z;
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

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdNormalInverseGauss<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.alpha, self.beta, self.delta, self.mu, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdNormalInverseGauss<T, R> {
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

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt
  for SimdNormalInverseGauss<T, R>
{
  fn pdf(&self, x: f64) -> f64 {
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
    a * d / std::f64::consts::PI
      * (d * gamma + b * (x - m) - a * q).exp()
      * crate::special::bessel_k1e(a * q)
      / q
  }

  fn cdf(&self, _x: f64) -> f64 {
    unimplemented!("DistributionExt::cdf for SimdNormalInverseGauss has no closed form")
  }

  fn inv_cdf(&self, _p: f64) -> f64 {
    unimplemented!("DistributionExt::inv_cdf for SimdNormalInverseGauss has no closed form")
  }

  fn mean(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let m = self.mu.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    m + d * b / gamma
  }

  fn median(&self) -> f64 {
    f64::NAN
  }

  fn mode(&self) -> f64 {
    // For NIG the mode is μ + δβ / sqrt(α² − β²) · (1 − ...) — no simple closed form.
    f64::NAN
  }

  fn variance(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    d * a * a / gamma.powi(3)
  }

  fn skewness(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    3.0 * b / (a * (d * gamma).sqrt())
  }

  fn kurtosis(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    3.0 * (1.0 + 4.0 * b * b / (a * a)) / (d * gamma)
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
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
    exponent.exp()
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    // M(t) = exp{ μt + δ (γ - sqrt(α² - (β + t)²)) }
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let m = self.mu.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    let bt = b + t;
    let inner = a * a - bt * bt;
    if inner < 0.0 {
      f64::INFINITY
    } else {
      (m * t + d * (gamma - inner.sqrt())).exp()
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
  use crate::traits::DistributionExt;

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
    let dist = SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.0, &Deterministic::new(1));
    let integral = trapezoid(-40.0, 40.0, 400_000, |x| dist.pdf(x));
    assert!(
      (integral - 1.0).abs() < 1e-6,
      "NIG pdf integral = {integral}, expected 1.0"
    );
  }

  /// First moment of pdf matches the existing closed-form mean() = μ + δβ/γ, tol 1e-5.
  #[test]
  fn nig_pdf_first_moment_matches_mean() {
    let dist = SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.0, &Deterministic::new(1));
    let first_moment = trapezoid(-40.0, 40.0, 400_000, |x| x * dist.pdf(x));
    let expected = dist.mean();
    assert!(
      (first_moment - expected).abs() < 1e-5,
      "first moment = {first_moment}, mean() = {expected}"
    );
  }

  #[test]
  fn nig_pdf_symmetric_when_beta_zero() {
    let mu = 0.5;
    let dist = SimdNormalInverseGauss::<f64>::new(2.0, 0.0, 1.0, mu, &Deterministic::new(1));
    for &h in &[0.1_f64, 0.5, 1.0, 2.0, 5.0] {
      let left = dist.pdf(mu - h);
      let right = dist.pdf(mu + h);
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
    let dist = SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.0, &Deterministic::new(1));
    let p = dist.pdf(2000.0);
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
    let dist = SimdNormalInverseGauss::<f64>::new(2.0, 1.9, 1.0, 0.0, &Deterministic::new(1));
    for i in 1..=200 {
      let x = i as f64 * 10.0;
      let p = dist.pdf(x);
      assert!(
        p.is_finite() && p >= 0.0,
        "pdf({x}) = {p}, expected finite >= 0"
      );
    }
  }

  /// `fill_slice`'s below-`SMALL_NIG_THRESHOLD` branch calls
  /// `ig.sample_fast()`/`normal.sample_fast()` directly (rather than the
  /// removed vestigial-`simd_rng`-backed `Distribution::sample(rng)` path);
  /// regression check that the refactor kept the same-seed replay
  /// contract and produced only finite output.
  #[test]
  fn nig_fill_slice_small_n_is_deterministic_and_finite() {
    let dist_a = SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.0, &Deterministic::new(7));
    let dist_b = SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.0, &Deterministic::new(7));
    let mut out_a = [0.0_f64; 8];
    let mut out_b = [0.0_f64; 8];
    dist_a.fill_slice(&mut out_a);
    dist_b.fill_slice(&mut out_b);
    assert_eq!(out_a, out_b, "same seed must replay bit-for-bit");
    assert!(
      out_a.iter().all(|x| x.is_finite()),
      "all samples must be finite, got {out_a:?}"
    );
  }
}
