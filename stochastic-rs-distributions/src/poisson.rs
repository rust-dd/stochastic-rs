//! # Poisson
//!
//! $$
//! \mathbb{P}(N=k)=e^{-\lambda}\frac{\lambda^k}{k!},\ k\in\mathbb N_0
//! $$
//!
//! Sampling: inversion on a cached cumulative table, Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §X.3, DOI 10.1007/978-1-4613-8643-8.

use std::marker::PhantomData;

use num_traits::PrimInt;
use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::seeded::StreamState;
use crate::source::uniform53;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Poisson law with rate `lambda`: parameters and the cumulative table; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Debug, PartialEq)]
pub struct SimdPoisson<T: PrimInt> {
  lambda: f64,
  cdf: Box<[f64]>,
  out: PhantomData<T>,
}

impl<T: PrimInt> SimdPoisson<T> {
  /// Builds the cumulative table from log-space pmf increments
  /// `ln pmf_k = -λ + k ln λ - ln Γ(k+1)`. The naive multiplicative
  /// recurrence starts at `exp(-λ)`, which underflows to 0 for λ ≳ 745 and
  /// then never recovers — the loop would run forever. The secondary stop
  /// condition covers accumulated-rounding cases where `cum` converges a
  /// few ulp below the `1 - 1e-15` target: past `2λ` with pmf < 4e-18 the
  /// remaining tail mass is below the table's own epsilon.
  #[inline]
  fn build_cdf(lambda: f64) -> Box<[f64]> {
    let mut cdf = Vec::new();
    let ln_lambda = lambda.ln();
    let mut log_pmf = -lambda;
    let mut cum = log_pmf.exp();
    cdf.push(cum);

    loop {
      let k = cdf.len() as f64;
      log_pmf += ln_lambda - k.ln();
      cum += log_pmf.exp();
      if cum >= 1.0 - 1e-15 || (k > 2.0 * lambda && log_pmf < -40.0) {
        cdf.push(1.0);
        break;
      }
      cdf.push(cum);
    }

    cdf.into_boxed_slice()
  }

  /// Creates a Poisson distribution.
  ///
  /// - `lambda` — finite rate λ > 0 (matches the module header's λ); mean and
  ///   variance are both λ. Stored at construction — see this type's
  ///   internal `build_cdf` for why the cumulative table must be built
  ///   in log space once λ ≳ 745.
  pub fn new(lambda: f64) -> Self {
    assert!(
      lambda > 0.0 && lambda.is_finite(),
      "lambda must satisfy `lambda > 0.0 && lambda.is_finite()`, got lambda = {lambda:?}"
    );
    Self {
      lambda,
      cdf: Self::build_cdf(lambda),
      out: PhantomData,
    }
  }

  /// The rate `λ`.
  pub fn lambda(&self) -> f64 {
    self.lambda
  }

  /// The count whose cumulative probability first reaches `u`; an overflow of `T` saturates (and asserts in debug).
  #[inline]
  fn index_of(&self, u: f64) -> T {
    let k = self.cdf.partition_point(|&p| p < u);
    let cast = num_traits::cast(k);
    debug_assert!(
      cast.is_some(),
      "Poisson draw {k} overflowed the output integer type"
    );
    cast.unwrap_or(T::max_value())
  }

  /// Kept out of line because inlined into a consumer's per-step pop the table search slows that loop by about 30 %.
  #[inline(never)]
  fn fill_parts<G: Rng + ?Sized>(&self, rng: &mut G, out: &mut [T]) {
    for x in out.iter_mut() {
      *x = self.index_of(uniform53(rng.next_u64()));
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.index_of(uniform53(rng.next_u64()))
  }
}

impl<T: PrimInt + Send + Sync + 'static> Sealed for SimdPoisson<T> {}

impl<T: PrimInt + Send + Sync + 'static> SimdDistribution for SimdPoisson<T> {
  type State<R: SimdRngExt> = StreamState<T, R, 16>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 16>, u64) {
    StreamState::init(seed)
  }
}

impl<T: PrimInt + Send + Sync + 'static> SimdKernel for SimdPoisson<T> {
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

impl<T: PrimInt + Send + Sync + 'static> Distribution<T> for SimdPoisson<T> {
  /// The table inversion of one 53-bit uniform from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: PrimInt> crate::traits::DistributionExt for SimdPoisson<T> {
  fn pdf(&self, x: f64) -> f64 {
    if x < 0.0 || x.fract() != 0.0 {
      return 0.0;
    }
    let k = x as i64;
    let lambda = self.lambda();
    // P(N=k) = exp(−λ) λ^k / k! = exp(k ln λ − λ − ln Γ(k+1))
    let log_pmf = k as f64 * lambda.ln() - lambda - crate::special::ln_gamma((k + 1) as f64);
    log_pmf.exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    if x < 0.0 {
      return 0.0;
    }
    let k = x.floor() as usize;
    if k >= self.cdf.len() {
      1.0
    } else {
      self.cdf[k]
    }
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    if p <= 0.0 {
      return 0.0;
    }
    if p >= 1.0 {
      return f64::INFINITY;
    }
    // Use the cached cumulative table built in `build_cdf`.
    match self.cdf.iter().position(|&c| c >= p) {
      Some(k) => k as f64,
      None => (self.cdf.len() - 1) as f64,
    }
  }

  fn mean(&self) -> f64 {
    self.lambda()
  }

  fn median(&self) -> f64 {
    // Approximation: ⌊λ + 1/3 - 0.02/λ⌋
    let l = self.lambda();
    (l + 1.0 / 3.0 - 0.02 / l).floor()
  }

  fn mode(&self) -> f64 {
    self.lambda().floor()
  }

  fn variance(&self) -> f64 {
    self.lambda()
  }

  fn skewness(&self) -> f64 {
    1.0 / self.lambda().sqrt()
  }

  fn kurtosis(&self) -> f64 {
    1.0 / self.lambda()
  }

  fn entropy(&self) -> f64 {
    // Closed form not elementary; fall back to an asymptotic expansion that's
    // accurate to leading order: H(λ) ≈ ½ ln(2π e λ) - 1/(12λ) - 1/(24λ²) - ...
    let l = self.lambda();
    0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * l).ln()
      - 1.0 / (12.0 * l)
      - 1.0 / (24.0 * l * l)
      - 19.0 / (360.0 * l.powi(3))
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    // φ(t) = exp(λ (e^{it} - 1))
    let eit = num_complex::Complex64::new(0.0, t).exp();
    (eit - num_complex::Complex64::new(1.0, 0.0))
      .scale(self.lambda())
      .exp()
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    (self.lambda() * (t.exp() - 1.0)).exp()
  }
}

py_distribution_int!(PyPoissonD, SimdPoisson,
  sig: (lambda_, seed=None),
  params: (lambda_: f64)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::SimdPoisson;
  use crate::tests::scalar_chi_square_best_p;
  use crate::traits::DistributionExt;
  use crate::traits::DistributionSampler;
  use crate::traits::SimdDistribution;

  /// λ ≳ 745 made the old multiplicative table build spin forever on an
  /// underflowed `exp(-λ)`; the log-space build must terminate and sample
  /// with the right mean.
  #[test]
  fn poisson_large_lambda_table_terminates() {
    let dist = SimdPoisson::<u64>::new(800.0);
    let mut buf = vec![0u64; 4096];
    dist
      .clone()
      .seeded(&Deterministic::new(3))
      .fill_slice(&mut buf);
    let mean = buf.iter().map(|&x| x as f64).sum::<f64>() / buf.len() as f64;
    assert!(
      (mean - 800.0).abs() < 3.0,
      "λ=800 sample mean drift: {mean}"
    );
    assert!((dist.mean() - 800.0).abs() < 1e-9);
  }

  /// An infinite rate turned the table build's log-pmf `NaN`, so neither exit fired and the table grew without bound.
  #[test]
  #[should_panic(
    expected = "lambda must satisfy `lambda > 0.0 && lambda.is_finite()`, got lambda = inf"
  )]
  fn poisson_infinite_lambda_is_rejected() {
    SimdPoisson::<u64>::new(f64::INFINITY);
  }

  /// Log-space build must reproduce the small-λ table semantics.
  #[test]
  fn poisson_small_lambda_moments() {
    let mut dist = SimdPoisson::<u32>::new(3.5).seeded(&Deterministic::new(11));
    let mut buf = vec![0u32; 100_000];
    dist.fill_slice(&mut buf);
    let n = buf.len() as f64;
    let mean = buf.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = buf
      .iter()
      .map(|&x| {
        let d = x as f64 - mean;
        d * d
      })
      .sum::<f64>()
      / n;
    assert!((mean - 3.5).abs() < 0.05, "mean drift: {mean}");
    assert!((var - 3.5).abs() < 0.15, "variance drift: {var}");
  }

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf, the log-space table included.
  #[test]
  fn scalar_sample_matches_cdf() {
    for (lambda, window) in [(12.0, (0, 40)), (800.0, (574, 1027))] {
      let d = SimdPoisson::<u64>::new(lambda);
      let best = scalar_chi_square_best_p(&d, window, |k| d.cdf(k as f64));
      assert!(best > 0.01, "Poisson({lambda}): best p = {best}");
    }
  }
}
