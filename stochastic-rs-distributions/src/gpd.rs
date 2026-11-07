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
//!   \mu + \dfrac{\sigma}{\xi}\,\mathrm{expm1}(-\xi\ln U), & \xi \neq 0 \\\\\[4pt\]
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

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::seeded::StreamState;
use crate::special::SHAPE_CUT;
use crate::special::ln_1p_over_xi;
use crate::special::pow_m1_over_xi;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_GPD_THRESHOLD: usize = 16;

/// Generalized Pareto distribution with location `μ`, scale `σ > 0` and
/// shape `ξ`; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdGpd<T> {
  mu: T,
  sigma: T,
  xi: T,
}

impl<T: SimdFloatExt> SimdGpd<T> {
  /// Construct a GPD$(\mu, \sigma, \xi)$.
  ///
  /// - `mu` — location μ, the lower end of the support.
  /// - `sigma` — scale σ > 0.
  /// - `xi` — shape ξ; positive for a heavy tail, zero for the exponential,
  ///   negative for a tail bounded at μ − σ/ξ.
  pub fn new(mu: T, sigma: T, xi: T) -> Self {
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(
      xi.is_finite(),
      "xi must satisfy `xi.is_finite()`, got xi = {xi:?}"
    );
    assert!(
      sigma > T::zero(),
      "sigma must satisfy `sigma > T::zero()`, got sigma = {sigma:?}"
    );
    Self { mu, sigma, xi }
  }

  /// The location `μ`.
  pub fn mu(&self) -> T {
    self.mu
  }

  /// The scale `σ`.
  pub fn sigma(&self) -> T {
    self.sigma
  }

  /// The shape `ξ`.
  pub fn xi(&self) -> T {
    self.xi
  }

  /// Clamp a uniform draw to the open unit interval so its logarithm stays finite at the lane level.
  #[inline]
  fn clamp_open_unit(x: T) -> T {
    let eps = T::from_f64_fast(1e-12);
    x.max(eps).min(T::one() - eps)
  }

  #[inline]
  fn is_exponential(&self) -> bool {
    self.xi.abs() < T::from_f64_fast(SHAPE_CUT)
  }

  /// The inverse cdf at `1 − u`, which is uniform when `u` is.
  #[inline]
  fn invert(&self, u: T, exponential: bool) -> T {
    let u = Self::clamp_open_unit(u);
    if exponential {
      self.mu - self.sigma * u.ln()
    } else {
      self.mu + self.sigma * pow_m1_over_xi(self.xi, u.ln())
    }
  }

  fn fill_parts<R: SimdRngExt>(&self, rng: &mut R, out: &mut [T]) {
    let exponential = self.is_exponential();
    if out.len() < SMALL_GPD_THRESHOLD {
      for x in out.iter_mut() {
        *x = self.invert(T::sample_uniform_simd(rng), exponential);
      }
      return;
    }
    let mu = T::splat(self.mu);
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
        // `σ` enters last: above the cut `σ/ξ` overflows once `σ > 1.8e289` (`3.4e19` in `f32`).
        let rise = T::simd_exp_m1(T::splat(-self.xi) * T::simd_ln(uv)) * T::splat(self.xi.recip());
        mu + T::splat(self.sigma) * rise
      };
      *chunk = T::simd_to_array(x);
    }
    for x in rem.iter_mut() {
      *x = self.invert(T::sample_uniform_simd(rng), exponential);
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.invert(T::sample_uniform(rng), self.is_exponential())
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

impl<T: SimdFloatExt> Sealed for SimdGpd<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdGpd<T> {
  type State<R: SimdRngExt> = StreamState<T, R, 16>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 16>, u64) {
    StreamState::init(seed)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdGpd<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>, out: &mut [T]) {
    self.fill_parts(&mut state.rng, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>) -> T {
    let StreamState { rng, buf } = state;
    buf.pop(|b| self.fill_parts(rng, b))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdGpd<T> {
  /// The inverse cdf at one `[0, 1)` uniform from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> DistributionExt for SimdGpd<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    let (mu, sigma, xi) = self.params();
    let z = (x - mu) / sigma;
    if z < 0.0 || z.is_infinite() || xi * z <= -1.0 {
      return Some(0.0);
    }
    Some((-(1.0 + xi) * ln_1p_over_xi(xi, z)).exp() / sigma)
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    let (mu, sigma, xi) = self.params();
    let z = (x - mu) / sigma;
    if z < 0.0 || z.is_infinite() || xi * z <= -1.0 {
      return Some(if z > 0.0 { 1.0 } else { 0.0 });
    }
    Some(-(-ln_1p_over_xi(xi, z)).exp_m1())
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) {
      return Some(f64::NAN);
    }
    if p == 0.0 {
      return Some(self.support().0);
    }
    if p == 1.0 {
      return Some(self.support().1);
    }
    let (mu, sigma, xi) = self.params();
    Some(mu + sigma * pow_m1_over_xi(xi, (-p).ln_1p()))
  }

  /// `+∞` for ξ ≥ 1, where the mean integral diverges.
  fn mean(&self) -> Option<f64> {
    let (mu, sigma, xi) = self.params();
    if xi < 1.0 {
      Some(mu + sigma / (1.0 - xi))
    } else {
      Some(f64::INFINITY)
    }
  }

  fn median(&self) -> Option<f64> {
    let (mu, sigma, xi) = self.params();
    Some(mu + sigma * pow_m1_over_xi(xi, -std::f64::consts::LN_2))
  }

  /// The density is decreasing for ξ > −1, so the mode sits at μ; below
  /// that it rises to the upper end of the support.
  fn mode(&self) -> Option<f64> {
    let (mu, sigma, xi) = self.params();
    if xi < -1.0 {
      Some(mu - sigma / xi)
    } else {
      Some(mu)
    }
  }

  /// `+∞` for ξ ≥ 1/2.
  fn variance(&self) -> Option<f64> {
    let (_, sigma, xi) = self.params();
    if xi < 0.5 {
      Some(sigma * sigma / ((1.0 - xi).powi(2) * (1.0 - 2.0 * xi)))
    } else {
      Some(f64::INFINITY)
    }
  }

  /// `NaN` for ξ ≥ 1/3, where the third moment is undefined.
  fn skewness(&self) -> Option<f64> {
    let (_, _, xi) = self.params();
    if xi < 1.0 / 3.0 {
      Some(2.0 * (1.0 + xi) * (1.0 - 2.0 * xi).sqrt() / (1.0 - 3.0 * xi))
    } else {
      Some(f64::NAN)
    }
  }

  /// Excess kurtosis; `NaN` for ξ ≥ 1/4.
  fn kurtosis(&self) -> Option<f64> {
    let (_, _, xi) = self.params();
    if xi < 0.25 {
      Some(
        3.0 * (1.0 - 2.0 * xi) * (2.0 * xi * xi + xi + 3.0) / ((1.0 - 3.0 * xi) * (1.0 - 4.0 * xi))
          - 3.0,
      )
    } else {
      Some(f64::NAN)
    }
  }

  fn entropy(&self) -> Option<f64> {
    let (_, sigma, xi) = self.params();
    Some(sigma.ln() + xi + 1.0)
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::tests::scalar_ks_best_p;
  use crate::traits::DistributionSampler;

  /// Exponential limit (ξ = 0): the sample mean of 30k draws must sit at σ.
  #[test]
  fn gpd_exponential_sample_mean() {
    let d = SimdGpd::<f64>::new(0.0, 2.0, 0.0);
    let n = 30_000;
    let mut xs = vec![0.0; n];
    d.seeded(&Deterministic::new(3)).fill_slice(&mut xs);
    let mean = xs.iter().sum::<f64>() / n as f64;
    assert!((mean - 2.0).abs() < 0.05, "mean = {mean}");
  }

  /// Bounded tail (ξ < 0): the sample mean matches μ + σ/(1−ξ) and no draw
  /// leaves the support.
  #[test]
  fn gpd_bounded_tail_moments_and_support() {
    let d = SimdGpd::<f64>::new(1.0, 1.0, -0.3);
    let n = 30_000;
    let mut xs = vec![0.0; n];
    d.seeded(&Deterministic::new(5)).fill_slice(&mut xs);
    let mean = xs.iter().sum::<f64>() / n as f64;
    assert!((mean - d.mean().unwrap()).abs() < 0.02, "mean = {mean}");
    let (lo, hi) = d.support();
    assert_eq!((lo, hi), (1.0, 1.0 + 1.0 / 0.3));
    assert!(xs.iter().all(|x| *x >= lo && *x <= hi + 1e-9));
  }

  #[test]
  fn scalar_sample_matches_cdf() {
    for xi in [0.0, 0.3, -0.3] {
      let d = SimdGpd::<f64>::new(0.0, 1.0, xi);
      let best = scalar_ks_best_p(&d, |x| d.cdf(x).unwrap());
      assert!(best > 0.01, "xi = {xi}: best p = {best}");
    }
  }

  /// PDF integrates to one over the support (midpoint rule) for a heavy
  /// tail.
  #[test]
  fn gpd_pdf_normalised() {
    let d = SimdGpd::<f64>::new(0.0, 1.0, 0.3);
    let n = 200_000usize;
    let up = 5_000.0_f64;
    let h = up / n as f64;
    let s: f64 = (0..n)
      .map(|k| d.pdf((k as f64 + 0.5) * h).unwrap() * h)
      .sum();
    assert!((s - 1.0).abs() < 2e-3, "PDF integrates to {s}");
  }

  /// F(F⁻¹(p)) = p for every shape sign.
  #[test]
  fn gpd_cdf_inverse_round_trip() {
    for xi in [0.3_f64, 0.0, -0.3] {
      let d = SimdGpd::<f64>::new(0.5, 2.0, xi);
      for p in [0.05_f64, 0.3, 0.5, 0.7, 0.95] {
        let x = d.quantile(p).unwrap();
        assert!(
          (d.cdf(x).unwrap() - p).abs() < 1e-12,
          "xi={xi}: F(F^-1({p})) = {}",
          d.cdf(x).unwrap()
        );
      }
      assert!((d.cdf(d.median().unwrap()).unwrap() - 0.5).abs() < 1e-12);
    }
  }

  #[test]
  fn gpd_deterministic_seed_reproduces_stream() {
    let mut a = SimdGpd::<f64>::new(0.5, 1.2, 0.3).seeded(&Deterministic::new(7));
    let mut b = SimdGpd::<f64>::new(0.5, 1.2, 0.3).seeded(&Deterministic::new(7));
    for _ in 0..256 {
      assert_eq!(a.sample(), b.sample());
    }
  }

  /// mpmath at 50 and 80 digits: `1 − t^{−1/ξ}` and `((1 − p)^{−ξ} − 1)/ξ` through `exp_m1` at tiny ξ.
  #[test]
  fn tiny_shape_keeps_its_digits() {
    let rel = |a: f64, b: f64| ((a - b) / b).abs();
    for (xi, q, c, median) in [
      (
        1e-12,
        2.302_585_092_996_697,
        0.864_664_716_763_116_6,
        0.693_147_180_560_185_5,
      ),
      (
        1e-9,
        2.302585095644995,
        0.864_664_716_492_716_7,
        0.693_147_180_800_171_8,
      ),
      (
        1e-7,
        2.302_585_358_088_972,
        0.864_664_689_696_331_6,
        0.693_147_204_582_596_6,
      ),
    ] {
      let g = SimdGpd::<f64>::new(0.0, 1.0, xi);
      assert!(
        rel(g.quantile(0.9).unwrap(), q) < 1e-13,
        "xi = {xi}: quantile"
      );
      assert!(rel(g.cdf(2.0).unwrap(), c) < 1e-13, "xi = {xi}: cdf");
      assert!(
        rel(g.median().unwrap(), median) < 1e-13,
        "xi = {xi}: median"
      );
    }
    let mut near = SimdGpd::<f64>::new(0.0, 1.0, 1e-11).seeded(&Deterministic::new(42));
    let mut exp = SimdGpd::<f64>::new(0.0, 1.0, 0.0).seeded(&Deterministic::new(42));
    for _ in 0..1_000 {
      let (a, b) = (near.sample(), exp.sample());
      assert!(
        (a - b).abs() < 1e-9,
        "ξ = 1e-11 draw {a} against the exponential draw {b}"
      );
    }
  }

  /// mpmath at 50 and 100 digits: where `ξ ln(1 − p)` or `ξz` leaves the normals, the forms keep the exponential's digits.
  #[test]
  fn subnormal_shape_products_keep_their_digits() {
    for xi in [1e-15, 1e-12] {
      assert_eq!(
        SimdGpd::<f64>::new(0.0, 1.0, xi).quantile(1e-300),
        Some(1e-300)
      );
    }
    assert_eq!(
      SimdGpd::<f64>::new(0.0, 1.0, 1e-4).quantile(3e-308),
      Some(3e-308)
    );
    assert_eq!(SimdGpd::<f64>::new(0.0, 1.0, 1e-310).pdf(0.0), Some(1.0));
  }

  /// A chunked fill with a scalar remainder, and a fill short enough to be all scalar.
  fn draws<T: SimdFloatExt>(sigma: T, xi: T) -> Vec<T> {
    let (mut long, mut short) = (vec![T::zero(); 4099], vec![T::zero(); 5]);
    let law = SimdGpd::new(T::zero(), sigma, xi);
    law.seeded(&Deterministic::new(42)).fill_slice(&mut long);
    law.seeded(&Deterministic::new(43)).fill_slice(&mut short);
    long.extend(short);
    long
  }

  /// Below the cut a stream draws the exponential's bits; above it no draw overflows at a scale the exponential keeps
  /// finite.
  #[test]
  fn tiny_shapes_draw_finite_values() {
    assert_eq!(draws(1000.0_f32, 1e-36), draws(1000.0_f32, 0.0));
    for (sigma, xi) in [(1.0, 1e-310), (1.0, 5e-324), (1e10, 1e-300)] {
      assert_eq!(
        draws(sigma, xi),
        draws(sigma, 0.0),
        "sigma = {sigma}, xi = {xi}"
      );
    }
    fn near<T: SimdFloatExt>(sigma: T, xi: T, tol: f64) {
      for (a, b) in draws(sigma, xi).into_iter().zip(draws(sigma, T::zero())) {
        let (a, b) = (a.to_f64().unwrap(), b.to_f64().unwrap());
        assert!(
          a.is_finite() && ((a - b) / b).abs() < tol,
          "sigma = {sigma:?}, xi = {xi:?}: {a} against {b}"
        );
      }
    }
    near(1e300, 2e-19, 1e-15);
    near(1e300, -1e-18, 1e-15);
    near(1e30_f32, 2e-19, 5e-7);
  }

  /// mpmath at 50 and 100 digits: the quantile at the tail levels next to ξ = 0 and at ξ = 0.2.
  #[test]
  fn quantile_at_the_tail_levels() {
    let levels = [1e-300, 1e-20, 1e-10, 1.0 - f64::EPSILON / 2.0, 1.0 - 1e-10];
    for (xi, upper) in [
      (
        1e-15,
        [1.00000000005e-10, 36.736800569677776, 23.025850847200353],
      ),
      (
        -1e-15,
        [1.00000000005e-10, 36.736800569676426, 23.025850847199823],
      ),
      (
        1e-8,
        [1.00000000005e-10, 36.736807317640505, 23.025853498149328],
      ),
      (
        -1e-8,
        [1.00000000005e-10, 36.736793821715345, 23.025848196251257],
      ),
      (
        0.2,
        [1.00000000006e-10, 7755.468820533241, 494.9999917259634],
      ),
    ] {
      let g = SimdGpd::<f64>::new(0.0, 1.0, xi);
      for (p, w) in levels
        .into_iter()
        .zip([1e-300, 1e-20].into_iter().chain(upper))
      {
        let q = g.quantile(p).unwrap();
        assert!(((q - w) / w).abs() < 1e-13, "xi = {xi:e}, p = {p:e}: {q:e}");
      }
    }
  }

  #[test]
  fn pdf_and_cdf_at_infinite_arguments() {
    for xi in [-2.0, -1.0, -0.5, 0.0, 0.5] {
      let g = SimdGpd::<f64>::new(0.0, 1.0, xi);
      assert_eq!(
        [f64::NEG_INFINITY, f64::INFINITY].map(|x| g.pdf(x)),
        [Some(0.0); 2],
        "xi = {xi}"
      );
      assert_eq!(
        [f64::NEG_INFINITY, f64::INFINITY].map(|x| g.cdf(x)),
        [Some(0.0), Some(1.0)],
        "xi = {xi}"
      );
    }
  }

  /// Moment existence thresholds: ξ = 0.3 keeps the mean and variance,
  /// loses the skewness and kurtosis.
  #[test]
  fn gpd_moment_thresholds() {
    let d = SimdGpd::<f64>::new(0.0, 1.0, 0.3);
    assert!(d.mean().unwrap().is_finite() && d.variance().unwrap().is_finite());
    assert!(d.skewness().unwrap().is_finite());
    assert!(d.kurtosis().unwrap().is_nan());
    let heavy = SimdGpd::<f64>::new(0.0, 1.0, 0.6);
    assert!(heavy.mean().unwrap().is_finite());
    assert_eq!(heavy.variance().unwrap(), f64::INFINITY);
    assert!(heavy.skewness().unwrap().is_nan());
  }
}

py_distribution!(PyGpd, SimdGpd,
  sig: (mu, sigma, xi, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, xi: f64)
);
