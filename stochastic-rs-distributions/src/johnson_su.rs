//! # Johnson SU
//!
//! Johnson's unbounded system: a normal deviate pushed through a shifted,
//! scaled hyperbolic sine,
//!
//! $$
//! X = \xi + \lambda\sinh\!\left(\frac{Z - \gamma}{\delta}\right),\qquad Z \sim \mathcal N(0, 1),
//! $$
//!
//! $$
//! f(x) = \frac{\delta}{\lambda\sqrt{2\pi}}\,\frac{1}{\sqrt{1 + y^2}}\,
//! \exp\!\Bigl[-\tfrac12\bigl(\gamma + \delta\,\operatorname{asinh} y\bigr)^2\Bigr],\qquad
//! F(x) = \Phi\bigl(\gamma + \delta\operatorname{asinh} y\bigr),\qquad y = \frac{x - \xi}{\lambda},
//! $$
//!
//! with shape $\gamma \in \mathbb R$ (skew), shape $\delta > 0$ (tail
//! weight), location $\xi$ and scale $\lambda > 0$. With
//! $\omega = e^{1/\delta^2}$ and $\Omega = \gamma/\delta$ the moments are
//! closed-form: mean $\xi - \lambda\sqrt\omega\sinh\Omega$, variance
//! $\tfrac12\lambda^2(\omega - 1)(\omega\cosh 2\Omega + 1)$, and the
//! third and fourth standardised moments of Johnson (1949). Every moment
//! exists for every parameter — the tails are lighter than any power law.
//!
//! Reference: Johnson, N.L. (1949), "Systems of Frequency Curves Generated
//! by Methods of Translation", *Biometrika* 36(1/2), 149-176.
//! DOI: 10.1093/biomet/36.1-2.149

use std::cell::Cell;
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use super::normal::SimdNormal;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::special::ndtri;
use crate::special::norm_cdf;

const SMALL_JSU_THRESHOLD: usize = 16;

/// Johnson SU distribution with shapes `γ`, `δ > 0`, location `ξ` and
/// scale `λ > 0`.
pub struct SimdJohnsonSu<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  gamma: T,
  delta: T,
  xi: T,
  lambda: T,
  normal: SimdNormal<T, 64, R>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdJohnsonSu<T, R> {
  /// Construct a Johnson SU$(\gamma, \delta, \xi, \lambda)$.
  pub fn new<S: crate::simd_rng::SeedExt>(gamma: T, delta: T, xi: T, lambda: T, seed: &S) -> Self {
    assert!(delta > T::zero(), "JohnsonSu: delta must be positive");
    assert!(lambda > T::zero(), "JohnsonSu: lambda must be positive");
    let normal = SimdNormal::<T, 64, R>::new(T::zero(), T::one(), seed);
    let stream_seed = seed.seed_value();
    Self {
      gamma,
      delta,
      xi,
      lambda,
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
      self.gamma,
      self.delta,
      self.xi,
      self.lambda,
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

  #[inline]
  fn transform(&self, z: T) -> T {
    let y = (z - self.gamma) / self.delta;
    self.xi + self.lambda * (y.exp() - (-y).exp()) / T::from_f64_fast(2.0)
  }

  /// Fills `out` with $\xi + \lambda\sinh((Z - \gamma)/\delta)$ from the
  /// internal normal stream, the hyperbolic sine evaluated 8-wide as
  /// $(e^y - e^{-y})/2$.
  pub fn fill_slice(&self, out: &mut [T]) {
    if out.len() < SMALL_JSU_THRESHOLD {
      for x in out.iter_mut() {
        *x = self.transform(self.normal.sample_fast());
      }
      return;
    }
    let gamma = T::splat(self.gamma);
    let inv_delta = T::splat(T::one() / self.delta);
    let xi = T::splat(self.xi);
    let half_lambda = T::splat(self.lambda / T::from_f64_fast(2.0));
    let zero = T::splat(T::zero());
    let mut zbuf = [T::zero(); 64];
    let (chunks, rem) = out.as_chunks_mut::<64>();
    for chunk in chunks {
      self.normal.fill_standard_fast(&mut zbuf);
      for (sub, z8) in chunk
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(zbuf.as_chunks::<8>().0.iter())
      {
        let y = (T::simd_from_array(*z8) - gamma) * inv_delta;
        let sinh = T::simd_exp(y) - T::simd_exp(zero - y);
        *sub = T::simd_to_array(xi + half_lambda * sinh);
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      self.normal.fill_standard_fast(&mut zbuf[..n]);
      for i in 0..n {
        rem[i] = self.transform(zbuf[i]);
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

  fn params(&self) -> (f64, f64, f64, f64) {
    (
      self.gamma.to_f64().unwrap(),
      self.delta.to_f64().unwrap(),
      self.xi.to_f64().unwrap(),
      self.lambda.to_f64().unwrap(),
    )
  }

  /// $(\omega, \Omega) = (e^{1/\delta^2}, \gamma/\delta)$.
  fn omegas(&self) -> (f64, f64) {
    let (gamma, delta, _, _) = self.params();
    ((1.0 / (delta * delta)).exp(), gamma / delta)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdJohnsonSu<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.gamma, self.delta, self.xi, self.lambda, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdJohnsonSu<T, R> {
  /// The `rng` argument is intentionally unused — this type draws from its
  /// own internal SIMD stream seeded at construction.
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

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdJohnsonSu<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let (gamma, delta, xi, lambda) = self.params();
    let y = (x - xi) / lambda;
    let z = gamma + delta * y.asinh();
    delta / (lambda * (2.0 * std::f64::consts::PI).sqrt()) / (1.0 + y * y).sqrt()
      * (-0.5 * z * z).exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    let (gamma, delta, xi, lambda) = self.params();
    norm_cdf(gamma + delta * ((x - xi) / lambda).asinh())
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let (gamma, delta, xi, lambda) = self.params();
    xi + lambda * ((ndtri(p) - gamma) / delta).sinh()
  }

  /// $\xi - \lambda\sqrt\omega\sinh\Omega$.
  fn mean(&self) -> f64 {
    let (_, _, xi, lambda) = self.params();
    let (omega, big) = self.omegas();
    xi - lambda * omega.sqrt() * big.sinh()
  }

  /// $\xi + \lambda\sinh(-\gamma/\delta)$.
  fn median(&self) -> f64 {
    let (gamma, delta, xi, lambda) = self.params();
    xi + lambda * (-gamma / delta).sinh()
  }

  /// $\tfrac12\lambda^2(\omega - 1)(\omega\cosh 2\Omega + 1)$.
  fn variance(&self) -> f64 {
    let (_, _, _, lambda) = self.params();
    let (omega, big) = self.omegas();
    0.5 * lambda * lambda * (omega - 1.0) * (omega * (2.0 * big).cosh() + 1.0)
  }

  /// $-\lambda^3\sqrt\omega(\omega-1)^2[\omega(\omega+2)\sinh 3\Omega + 3\sinh\Omega]/(4\,\mathrm{Var}^{3/2})$.
  fn skewness(&self) -> f64 {
    let (_, _, _, lambda) = self.params();
    let (omega, big) = self.omegas();
    let var = self.variance();
    -lambda.powi(3)
      * omega.sqrt()
      * (omega - 1.0).powi(2)
      * (omega * (omega + 2.0) * (3.0 * big).sinh() + 3.0 * big.sinh())
      / (4.0 * var.powf(1.5))
  }

  /// Excess kurtosis
  /// $\lambda^4(\omega-1)^2[\omega^2(\omega^4 + 2\omega^3 + 3\omega^2 - 3)\cosh 4\Omega + 4\omega^2(\omega+2)\cosh 2\Omega + 3(2\omega+1)]/(8\,\mathrm{Var}^2) - 3$.
  fn kurtosis(&self) -> f64 {
    let (_, _, _, lambda) = self.params();
    let (omega, big) = self.omegas();
    let var = self.variance();
    let w2 = omega * omega;
    let bracket = w2 * (w2 * w2 + 2.0 * omega.powi(3) + 3.0 * w2 - 3.0) * (4.0 * big).cosh()
      + 4.0 * w2 * (omega + 2.0) * (2.0 * big).cosh()
      + 3.0 * (2.0 * omega + 1.0);
    lambda.powi(4) * (omega - 1.0).powi(2) * bracket / (8.0 * var * var) - 3.0
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::traits::DistributionExt;

  fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1e-300)
  }

  /// `scipy.stats.johnsonsu(a=γ, b=δ, loc=ξ, scale=λ)`: pdf / cdf on a grid
  /// and `stats('mvsk')`, `median`, `ppf`. The pdf and moments agree to
  /// 1e-12; the cdf and quantile inherit the crate's `erf` (A&S 7.1.26,
  /// 1.5e-7) and `ndtri` (Acklam, 1.15e-9) accuracy, so they are held at
  /// 3e-6 and 1e-7 respectively.
  #[test]
  fn matches_scipy_johnsonsu() {
    let cases: [((f64, f64, f64, f64), [(f64, f64, f64); 6], [f64; 7]); 2] = [
      (
        (-0.5, 1.5, 0.2, 2.0),
        [
          (-3.0, 0.009_483_587_115_503_317, 0.008_810_793_154_390_135),
          (-1.0, 0.102_695_332_018_599_79, 0.087_989_908_627_922_22),
          (0.0, 0.241_066_119_347_473_35, 0.257_926_499_201_360_9),
          (0.5, 0.284_851_162_511_866_6, 0.391_337_277_230_938_9),
          (2.0, 0.172_443_995_137_252_45, 0.762_170_056_063_578_9),
          (6.0, 0.009_071_300_360_203_717, 0.985_353_109_816_126_1),
        ],
        [
          1.048_069_681_819_088,
          3.267_344_543_424_031_6,
          0.999_035_241_982_614_7,
          5.379_444_246_948_418_5,
          0.879_081_114_512_300_3,
          3.174_611_571_098_172,
          -1.479_050_736_235_289_2,
        ],
      ),
      (
        (1.0, 0.8, -1.0, 0.5),
        [
          (-3.0, 0.123_208_880_563_184_71, 0.249_593_330_314_695_1),
          (-1.0, 0.387_153_159_230_629_43, 0.841_344_746_068_542_9),
          (0.0, 0.028_002_291_760_099_7, 0.984_415_497_757_174_9),
          (0.5, 0.009_920_580_652_585_82, 0.992_951_010_715_771_2),
          (2.0, 0.001_188_947_128_701_483_8, 0.998_620_668_438_965_7),
          (6.0, 5.472_542_541_178_683_4e-5, 0.999_877_189_022_907_6),
        ],
        [
          -2.749_456_477_027_036,
          14.260_685_190_023_532,
          -12.455_716_098_287_539,
          737.547_610_504_787_9,
          -1.800_959_540_150_412_8,
          -0.820_375_065_596_033_5,
          -7.810_243_036_204_189,
        ],
      ),
    ];
    for ((g, d, xi, lam), grid, stats) in cases {
      let dist = SimdJohnsonSu::<f64>::new(g, d, xi, lam, &Unseeded);
      for (x, pdf, cdf) in grid {
        assert!(close(dist.pdf(x), pdf, 1e-12), "pdf({x}) = {}", dist.pdf(x));
        assert!(close(dist.cdf(x), cdf, 3e-6), "cdf({x}) = {}", dist.cdf(x));
      }
      assert!(close(dist.mean(), stats[0], 1e-12), "mean {}", dist.mean());
      assert!(
        close(dist.variance(), stats[1], 1e-12),
        "variance {}",
        dist.variance()
      );
      assert!(
        close(dist.skewness(), stats[2], 1e-11),
        "skewness {}",
        dist.skewness()
      );
      assert!(
        close(dist.kurtosis(), stats[3], 1e-11),
        "kurtosis {}",
        dist.kurtosis()
      );
      assert!(
        close(dist.median(), stats[4], 1e-12),
        "median {}",
        dist.median()
      );
      assert!(
        close(dist.inv_cdf(0.9), stats[5], 1e-7),
        "ppf(0.9) {}",
        dist.inv_cdf(0.9)
      );
      assert!(
        close(dist.inv_cdf(0.05), stats[6], 1e-7),
        "ppf(0.05) {}",
        dist.inv_cdf(0.05)
      );
      for p in [0.01, 0.3, 0.5, 0.8, 0.99] {
        assert!((dist.cdf(dist.inv_cdf(p)) - p).abs() < 1e-6);
      }
    }
  }

  /// The SIMD sinh transform reproduces the closed-form mean and variance.
  #[test]
  fn sample_moments_match_closed_forms() {
    let dist = SimdJohnsonSu::<f64>::new(-0.5, 1.5, 0.2, 2.0, &Deterministic::new(5));
    let n = 200_000;
    let mut xs = vec![0.0; n];
    dist.fill_slice(&mut xs);
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    assert!((mean - dist.mean()).abs() < 0.02, "mean {mean}");
    assert!(
      (var - dist.variance()).abs() / dist.variance() < 0.03,
      "var {var}"
    );
  }

  #[test]
  fn deterministic_seed_reproduces_stream() {
    let a = SimdJohnsonSu::<f64>::new(1.0, 0.8, -1.0, 0.5, &Deterministic::new(7));
    let b = SimdJohnsonSu::<f64>::new(1.0, 0.8, -1.0, 0.5, &Deterministic::new(7));
    for _ in 0..256 {
      assert_eq!(a.sample_fast(), b.sample_fast());
    }
  }
}

py_distribution!(PyJohnsonSu, SimdJohnsonSu,
  sig: (gamma, delta, xi, lambda, seed=None, dtype=None),
  params: (gamma: f64, delta: f64, xi: f64, lambda: f64)
);
