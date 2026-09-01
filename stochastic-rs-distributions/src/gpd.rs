//! # Generalized Pareto distribution (GPD)
//!
//! The limit law of excesses over a high threshold (Pickands 1975,
//! Balkema–de Haan 1974), the peaks-over-threshold counterpart of the GEV:
//!
//! $$
//! F(x;\mu,\sigma,\xi) =
//! \begin{cases}
//!   1 - \bigl(1 + \xi\,\tfrac{x - \mu}{\sigma}\bigr)^{-1/\xi}, & \xi \neq 0 \\\\\[4pt\]
//!   1 - \exp\!\bigl(-(x-\mu)/\sigma\bigr), & \xi = 0
//! \end{cases}
//! $$
//!
//! on $x \ge \mu$, and additionally $x \le \mu - \sigma/\xi$ when $\xi < 0$,
//! with $\mu \in \mathbb R$ (location), $\sigma > 0$ (scale) and $\xi \in
//! \mathbb R$ (shape): $\xi > 0$ is a Pareto-type heavy tail, $\xi = 0$ the
//! exponential, $\xi < 0$ a bounded tail. Moments exist up to order
//! $1/\xi$: the mean needs $\xi < 1$, the variance $\xi < 1/2$, the
//! skewness $\xi < 1/3$ and the excess kurtosis $\xi < 1/4$.
//!
//! ## Sampling
//!
//! Closed-form inverse CDF on the internal SIMD stream:
//!
//! $$
//! X = \begin{cases}
//!   \mu + \dfrac{\sigma}{\xi}\bigl(U^{-\xi} - 1\bigr), & \xi \neq 0 \\\\\[4pt\]
//!   \mu - \sigma\,\ln U, & \xi = 0
//! \end{cases},
//! \qquad U \sim \mathrm{Uniform}(0, 1).
//! $$
//!
//! References:
//! - Pickands, J. (1975), "Statistical Inference Using Extreme Order
//!   Statistics", *Annals of Statistics* 3(1), 119-131.
//!   DOI: 10.1214/aos/1176343003
//! - Balkema, A.A., de Haan, L. (1974), "Residual Life Time at Great Age",
//!   *Annals of Probability* 2(5), 792-804. DOI: 10.1214/aop/1176996548
//! - Hosking, J.R.M., Wallis, J.R. (1987), "Parameter and Quantile
//!   Estimation for the Generalized Pareto Distribution", *Technometrics*
//!   29(3), 339-349 (moments). DOI: 10.1080/00401706.1987.10488243
//! - Coles, S. (2001), *An Introduction to Statistical Modeling of Extreme
//!   Values*, Springer, ch. 4.

use std::cell::Cell;
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;

const SMALL_GPD_THRESHOLD: usize = 16;

/// Generalized Pareto distribution with location `μ`, scale `σ > 0` and
/// shape `ξ`.
///
/// Sampling uses the closed-form inverse CDF from the module docs on the
/// internal SIMD RNG — bulk fills vectorise the `ln` / `powf` chain 8-wide.
pub struct SimdGpd<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  mu: T,
  sigma: T,
  xi: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
  stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdGpd<T, R> {
  /// Construct a GPD$(\mu, \sigma, \xi)$.
  ///
  /// - `mu` — location μ, the lower end of the support.
  /// - `sigma` — scale σ > 0.
  /// - `xi` — shape ξ; positive for a heavy tail, zero for the exponential,
  ///   negative for a tail bounded at μ − σ/ξ.
  pub fn new<S: SeedExt>(mu: T, sigma: T, xi: T, seed: &S) -> Self {
    assert!(sigma > T::zero(), "σ must be positive");
    let stream_seed = seed.seed_value();
    Self {
      mu,
      sigma,
      xi,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
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
      self.mu,
      self.sigma,
      self.xi,
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

  /// Clamp a uniform draw to the open unit interval so the `ln` / `powf`
  /// chain stays finite at the lane level.
  #[inline]
  fn clamp_open_unit(x: T) -> T {
    let eps = T::from_f64_fast(1e-12);
    x.max(eps).min(T::one() - eps)
  }

  /// One inverse-CDF draw on the internal RNG.
  #[inline]
  fn sample_one(&self, rng: &mut R, exponential: bool) -> T {
    let u = Self::clamp_open_unit(T::sample_uniform_simd(rng));
    if exponential {
      self.mu - self.sigma * u.ln()
    } else {
      self.mu + (self.sigma / self.xi) * (u.powf(-self.xi) - T::one())
    }
  }

  /// Fills `out` with GPD samples using the internal SIMD RNG stream — the
  /// only stream this sampler draws from (see the crate-level RNG policy).
  /// The inverse-CDF transform runs 8-wide.
  pub fn fill_slice(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let exponential = self.xi.to_f64().unwrap().abs() < 1e-12;
    if out.len() < SMALL_GPD_THRESHOLD {
      for x in out.iter_mut() {
        *x = self.sample_one(rng, exponential);
      }
      return;
    }
    let mu = T::splat(self.mu);
    let one = T::splat(T::one());
    let mut u = [T::zero(); 8];
    let (chunks, rem) = out.as_chunks_mut::<8>();
    for chunk in chunks {
      T::fill_uniform_simd(rng, &mut u);
      for x in u.iter_mut() {
        *x = Self::clamp_open_unit(*x);
      }
      let uv = T::simd_from_array(u);
      let x = if exponential {
        mu - T::splat(self.sigma) * T::simd_ln(uv)
      } else {
        mu + T::splat(self.sigma / self.xi) * (T::simd_powf(uv, -self.xi) - one)
      };
      *chunk = T::simd_to_array(x);
    }
    for x in rem.iter_mut() {
      *x = self.sample_one(rng, exponential);
    }
  }

  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice(buf);
    unsafe {
      *self.index.get() = 0;
    }
  }

  /// Support as `(lo, hi)`: `[μ, ∞)` for ξ ≥ 0, `[μ, μ − σ/ξ]` for ξ < 0.
  pub fn support(&self) -> (f64, f64) {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    let xi = self.xi.to_f64().unwrap();
    if xi < 0.0 {
      (mu, mu - sigma / xi)
    } else {
      (mu, f64::INFINITY)
    }
  }

  fn params(&self) -> (f64, f64, f64) {
    (
      self.mu.to_f64().unwrap(),
      self.sigma.to_f64().unwrap(),
      self.xi.to_f64().unwrap(),
    )
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdGpd<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.mu, self.sigma, self.xi, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdGpd<T, R> {
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

impl<T: SimdFloatExt, R: SimdRngExt> DistributionExt for SimdGpd<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let (mu, sigma, xi) = self.params();
    let z = (x - mu) / sigma;
    if z < 0.0 {
      return 0.0;
    }
    if xi.abs() < 1e-12 {
      (-z).exp() / sigma
    } else {
      let t = 1.0 + xi * z;
      if t <= 0.0 {
        return 0.0;
      }
      t.powf(-1.0 / xi - 1.0) / sigma
    }
  }

  fn cdf(&self, x: f64) -> f64 {
    let (mu, sigma, xi) = self.params();
    let z = (x - mu) / sigma;
    if z < 0.0 {
      return 0.0;
    }
    if xi.abs() < 1e-12 {
      1.0 - (-z).exp()
    } else {
      let t = 1.0 + xi * z;
      if t <= 0.0 {
        return 1.0;
      }
      1.0 - t.powf(-1.0 / xi)
    }
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let (mu, sigma, xi) = self.params();
    if xi.abs() < 1e-12 {
      mu - sigma * (1.0 - p).ln()
    } else {
      mu + sigma / xi * ((1.0 - p).powf(-xi) - 1.0)
    }
  }

  /// `+∞` for ξ ≥ 1, where the mean integral diverges.
  fn mean(&self) -> f64 {
    let (mu, sigma, xi) = self.params();
    if xi < 1.0 {
      mu + sigma / (1.0 - xi)
    } else {
      f64::INFINITY
    }
  }

  fn median(&self) -> f64 {
    let (mu, sigma, xi) = self.params();
    if xi.abs() < 1e-12 {
      mu + sigma * std::f64::consts::LN_2
    } else {
      mu + sigma * (2.0_f64.powf(xi) - 1.0) / xi
    }
  }

  /// The density is decreasing for ξ > −1, so the mode sits at μ; below
  /// that it rises to the upper end of the support.
  fn mode(&self) -> f64 {
    let (mu, sigma, xi) = self.params();
    if xi < -1.0 { mu - sigma / xi } else { mu }
  }

  /// `+∞` for ξ ≥ 1/2.
  fn variance(&self) -> f64 {
    let (_, sigma, xi) = self.params();
    if xi < 0.5 {
      sigma * sigma / ((1.0 - xi).powi(2) * (1.0 - 2.0 * xi))
    } else {
      f64::INFINITY
    }
  }

  /// `NaN` for ξ ≥ 1/3, where the third moment is undefined.
  fn skewness(&self) -> f64 {
    let (_, _, xi) = self.params();
    if xi < 1.0 / 3.0 {
      2.0 * (1.0 + xi) * (1.0 - 2.0 * xi).sqrt() / (1.0 - 3.0 * xi)
    } else {
      f64::NAN
    }
  }

  /// Excess kurtosis; `NaN` for ξ ≥ 1/4.
  fn kurtosis(&self) -> f64 {
    let (_, _, xi) = self.params();
    if xi < 0.25 {
      3.0 * (1.0 - 2.0 * xi) * (2.0 * xi * xi + xi + 3.0) / ((1.0 - 3.0 * xi) * (1.0 - 4.0 * xi))
        - 3.0
    } else {
      f64::NAN
    }
  }

  fn entropy(&self) -> f64 {
    let (_, sigma, xi) = self.params();
    sigma.ln() + xi + 1.0
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// Exponential limit (ξ = 0): the sample mean of 30k draws must sit at σ.
  #[test]
  fn gpd_exponential_sample_mean() {
    let d = SimdGpd::<f64>::new(0.0, 2.0, 0.0, &Deterministic::new(3));
    let n = 30_000;
    let mut xs = vec![0.0; n];
    d.fill_slice(&mut xs);
    let mean = xs.iter().sum::<f64>() / n as f64;
    assert!((mean - 2.0).abs() < 0.05, "mean = {mean}");
  }

  /// Bounded tail (ξ < 0): the sample mean matches μ + σ/(1−ξ) and no draw
  /// leaves the support.
  #[test]
  fn gpd_bounded_tail_moments_and_support() {
    let d = SimdGpd::<f64>::new(1.0, 1.0, -0.3, &Deterministic::new(5));
    let n = 30_000;
    let mut xs = vec![0.0; n];
    d.fill_slice(&mut xs);
    let mean = xs.iter().sum::<f64>() / n as f64;
    assert!((mean - d.mean()).abs() < 0.02, "mean = {mean}");
    let (lo, hi) = d.support();
    assert_eq!((lo, hi), (1.0, 1.0 + 1.0 / 0.3));
    assert!(xs.iter().all(|x| *x >= lo && *x <= hi + 1e-9));
  }

  /// PDF integrates to one over the support (midpoint rule) for a heavy
  /// tail.
  #[test]
  fn gpd_pdf_normalised() {
    let d = SimdGpd::<f64>::new(0.0, 1.0, 0.3, &Unseeded);
    let n = 200_000usize;
    let up = 5_000.0_f64;
    let h = up / n as f64;
    let s: f64 = (0..n).map(|k| d.pdf((k as f64 + 0.5) * h) * h).sum();
    assert!((s - 1.0).abs() < 2e-3, "PDF integrates to {s}");
  }

  /// F(F⁻¹(p)) = p for every shape sign.
  #[test]
  fn gpd_cdf_inverse_round_trip() {
    for xi in [0.3_f64, 0.0, -0.3] {
      let d = SimdGpd::<f64>::new(0.5, 2.0, xi, &Unseeded);
      for p in [0.05_f64, 0.3, 0.5, 0.7, 0.95] {
        let x = d.inv_cdf(p);
        assert!(
          (d.cdf(x) - p).abs() < 1e-12,
          "xi={xi}: F(F^-1({p})) = {}",
          d.cdf(x)
        );
      }
      assert!((d.cdf(d.median()) - 0.5).abs() < 1e-12);
    }
  }

  #[test]
  fn gpd_deterministic_seed_reproduces_stream() {
    let a = SimdGpd::<f64>::new(0.5, 1.2, 0.3, &Deterministic::new(7));
    let b = SimdGpd::<f64>::new(0.5, 1.2, 0.3, &Deterministic::new(7));
    for _ in 0..256 {
      assert_eq!(a.sample_fast(), b.sample_fast());
    }
  }

  /// Moment existence thresholds: ξ = 0.3 keeps the mean and variance,
  /// loses the skewness and kurtosis.
  #[test]
  fn gpd_moment_thresholds() {
    let d = SimdGpd::<f64>::new(0.0, 1.0, 0.3, &Unseeded);
    assert!(d.mean().is_finite() && d.variance().is_finite());
    assert!(d.skewness().is_finite());
    assert!(d.kurtosis().is_nan());
    let heavy = SimdGpd::<f64>::new(0.0, 1.0, 0.6, &Unseeded);
    assert!(heavy.mean().is_finite());
    assert_eq!(heavy.variance(), f64::INFINITY);
    assert!(heavy.skewness().is_nan());
  }
}

py_distribution!(PyGpd, SimdGpd,
  sig: (mu, sigma, xi, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, xi: f64)
);
