//! # Poisson
//!
//! $$
//! \mathbb{P}(N=k)=e^{-\lambda}\frac{\lambda^k}{k!},\ k\in\mathbb N_0
//! $$
//!
//! Sampling: table inversion below λ = 1000, Hörmann's PTRS from it on.
//!
//! Reference: Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §X.3, DOI 10.1007/978-1-4613-8643-8.
//! Reference: Hörmann, W. (1993), "The transformed rejection method for generating Poisson random variables", *Insurance: Mathematics and Economics* 12(1), 39-45, DOI 10.1016/0167-6687(93)90997-4.
//! Reference: Loader, C. (2000), "Fast and Accurate Computation of Binomial Probabilities", technical report, <https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf> (the saddle-point `ln pₖ`).
//! Reference: Choi, K.P. (1994), "On the medians of gamma distributions and an equation of Ramanujan", *Proceedings of the American Mathematical Society* 121(1), 245-251, DOI 10.1090/S0002-9939-1994-1195477-8 (the median bracket `λ − ln 2 ≤ m < λ + 1/3`).

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use num_traits::PrimInt;
use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::seeded::StreamState;
use crate::source::uniform53;
use crate::special::HALF_LN_TAU;
use crate::special::gamma_q;
use crate::special::ln_gamma;
use crate::special::ndtri;
use crate::special::stirling_tail;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Rates from here on draw by PTRS (Hörmann's own floor is 10); below it the table is exact inversion and holds ≤ 1 300 entries.
const PTRS_MIN_LAMBDA: f64 = 1000.0;

/// From this count on Stirling's series holds `stirlerr(k)` within `5e-18`; below it `ln pₖ` is `k ln λ − λ − ln k!`.
const STIRLING_FROM: f64 = 15.0;

/// `ln pₖ` as `−stirlerr(k) − bd0(k, λ) − ½ ln 2πk` from `STIRLING_FROM` on, which keeps the digits `k ln λ − λ − ln k!` cancels at large λ.
fn ln_pmf(k: f64, lambda: f64, ln_lambda: f64) -> f64 {
  if k < STIRLING_FROM {
    return if k == 0.0 {
      -lambda
    } else {
      k * ln_lambda - lambda - ln_gamma(k + 1.0)
    };
  }
  -stirling_tail(k.recip()) - bd0(k, lambda, ln_lambda) - HALF_LN_TAU.0 - 0.5 * k.ln()
}

/// `k ln(k/λ) + λ − k`: Loader's series in `v = (k − λ)/(k + λ)` for `|v| < ½`, whose terms do not cancel, the logarithm form outside.
fn bd0(k: f64, lambda: f64, ln_lambda: f64) -> f64 {
  let d = k - lambda;
  let v = (0.5 * d) / (0.5 * k + 0.5 * lambda);
  if v.abs() < 0.5 {
    let v2 = v * v;
    let (mut s, mut term) = (d * v, 2.0 * (k * v));
    let mut j = 1.0;
    while j < 64.0 {
      term *= v2;
      let next = s + term / (2.0 * j + 1.0);
      if next == s {
        break;
      }
      s = next;
      j += 1.0;
    }
    return s;
  }
  let ratio = k / lambda;
  let ln_ratio = if ratio.is_finite() {
    ratio.ln()
  } else {
    k.ln() - ln_lambda
  };
  k * ln_ratio + lambda - k
}

/// Hörmann (1993), algorithm PTRS: the set-up constants of one rate.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ptrs {
  mu: f64,
  ln_mu: f64,
  b: f64,
  a: f64,
  ln_inv_alpha: f64,
  v_r: f64,
}

impl Ptrs {
  fn new(mu: f64) -> Self {
    let b = 0.931 + 2.53 * mu.sqrt();
    let a = -0.059 + 0.02483 * b;
    Self {
      mu,
      ln_mu: mu.ln(),
      b,
      a,
      ln_inv_alpha: (1.1239 + 1.1328 / (b - 3.4)).ln(),
      v_r: 0.9277 - 3.6224 / (b - 2.0),
    }
  }

  /// Two uniforms per attempt, 88 % accepted from λ = 1000 on; the squeeze accepts without the logarithm.
  #[inline]
  fn draw(&self, mut uniform: impl FnMut() -> f64) -> f64 {
    loop {
      let u = uniform() - 0.5;
      let v = uniform();
      let us = 0.5 - u.abs();
      let k = ((2.0 * self.a / us + self.b) * u + self.mu + 0.43).floor();
      if us >= 0.07 && v <= self.v_r {
        return k;
      }
      if k < 0.0 || (us < 0.013 && v > us) {
        continue;
      }
      let lhs = v.ln() + self.ln_inv_alpha - (self.a / (us * us) + self.b).ln();
      if lhs <= ln_pmf(k, self.mu, self.ln_mu) {
        return k;
      }
    }
  }

  /// [`Self::draw`] on `rng`, out of line so that a table law's scalar draw stays small enough to inline.
  #[inline(never)]
  fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> f64 {
    self.draw(|| uniform53(rng.next_u64()))
  }
}

#[inline]
fn table_index(table: &[f64], u: f64) -> u64 {
  table.partition_point(|&c| c < u) as u64
}

/// Visits the table's entries `P(N ≤ k)` from `k = 0`, in log space as `exp(−λ)` underflows at λ ≳ 745, until `visit` answers `true` or the table ends; returns that `k`.
#[inline]
fn walk(lambda: f64, mut visit: impl FnMut(f64) -> bool) -> f64 {
  let ln_lambda = lambda.ln();
  let mut log_pmf = -lambda;
  let mut cum = log_pmf.exp();
  let mut k = 0.0_f64;
  loop {
    if visit(cum) {
      return k;
    }
    k += 1.0;
    log_pmf += ln_lambda - k.ln();
    cum += log_pmf.exp();
    // Past 2λ the sum can settle a few ulp below its target.
    if cum >= 1.0 - 1e-15 || (k > 2.0 * lambda && log_pmf < -40.0) {
      visit(1.0);
      return k;
    }
  }
}

fn walk_count(lambda: f64, u: f64) -> f64 {
  walk(lambda, |cum| cum >= u)
}

/// The count a `SimdPoisson` of `lambda` draws from `rng`, as an `f64` that keeps counts past `u64`: [`walk_count`] below `PTRS_MIN_LAMBDA`, PTRS from it on.
pub(crate) fn count_once<G: Rng + ?Sized>(lambda: f64, rng: &mut G) -> f64 {
  if lambda < PTRS_MIN_LAMBDA {
    walk_count(lambda, uniform53(rng.next_u64()))
  } else {
    Ptrs::new(lambda).draw(|| uniform53(rng.next_u64()))
  }
}

#[derive(Clone, Debug, PartialEq)]
enum Sampler {
  Table(Box<[f64]>),
  Ptrs(Ptrs),
}

/// Poisson law with rate `lambda`: the rate and its sampler (a table below λ = 1000, PTRS's constants from it on); a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Debug, PartialEq)]
pub struct SimdPoisson<T: PrimInt> {
  lambda: f64,
  sampler: Sampler,
  out: PhantomData<T>,
}

impl<T: PrimInt> SimdPoisson<T> {
  #[inline]
  fn build_cdf(lambda: f64) -> Box<[f64]> {
    let mut cdf = Vec::new();
    walk(lambda, |cum| {
      cdf.push(cum);
      false
    });
    cdf.into_boxed_slice()
  }

  /// The law of rate `lambda`, its mean and variance; panics unless `0 < lambda < ∞`.
  pub fn new(lambda: f64) -> Self {
    assert!(
      lambda > 0.0 && lambda.is_finite(),
      "lambda must satisfy `lambda > 0.0 && lambda.is_finite()`, got lambda = {lambda:?}"
    );
    let sampler = if lambda < PTRS_MIN_LAMBDA {
      Sampler::Table(Self::build_cdf(lambda))
    } else {
      Sampler::Ptrs(Ptrs::new(lambda))
    };
    Self {
      lambda,
      sampler,
      out: PhantomData,
    }
  }

  /// The rate `λ`.
  pub fn lambda(&self) -> f64 {
    self.lambda
  }

  /// `k` as `T`, saturated past `T`'s range (an assertion in debug); PTRS's `f64` count saturates at `u64::MAX` first.
  #[inline]
  fn count(k: u64) -> T {
    let cast = num_traits::cast(k);
    debug_assert!(
      cast.is_some(),
      "Poisson draw {k} overflowed the output integer type"
    );
    cast.unwrap_or(T::max_value())
  }

  /// Kept out of line because inlined into a consumer's per-step pop the table search slows that loop by about 30 %.
  #[inline(never)]
  fn fill_parts<G: Rng + ?Sized>(&self, rng: &mut G, out: &mut [MaybeUninit<T>]) {
    match &self.sampler {
      Sampler::Table(table) => {
        for x in out.iter_mut() {
          x.write(Self::count(table_index(table, uniform53(rng.next_u64()))));
        }
      }
      Sampler::Ptrs(ptrs) => {
        for x in out.iter_mut() {
          x.write(Self::count(ptrs.draw(|| uniform53(rng.next_u64())) as u64));
        }
      }
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    match &self.sampler {
      Sampler::Table(table) => Self::count(table_index(table, uniform53(rng.next_u64()))),
      Sampler::Ptrs(ptrs) => Self::count(ptrs.draw_with(rng) as u64),
    }
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
  fn fill<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>, out: &mut [MaybeUninit<T>]) {
    self.fill_parts(&mut state.rng, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>) -> T {
    let StreamState { rng, buf } = state;
    buf.pop(|b| self.fill_parts(rng, b))
  }
}

impl<T: PrimInt + Send + Sync + 'static> Distribution<T> for SimdPoisson<T> {
  /// One draw on the caller's rng: the table inversion of a 53-bit uniform below λ = 1000, PTRS from it on.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: PrimInt> crate::traits::DistributionExt for SimdPoisson<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    if x.is_nan() {
      return Some(f64::NAN);
    }
    if x < 0.0 || x.fract() != 0.0 {
      return Some(0.0);
    }
    Some(ln_pmf(x, self.lambda, self.lambda.ln()).exp())
  }

  /// `Q(⌊x⌋ + 1, λ)`: the regularised upper incomplete gamma, no table; below 1 it is `e^{−λ}` itself, which `gamma_q(1, λ)` reads ulps off.
  fn cdf(&self, x: f64) -> Option<f64> {
    if x.is_nan() {
      return Some(f64::NAN);
    }
    if x < 0.0 {
      return Some(0.0);
    }
    if x == f64::INFINITY {
      return Some(1.0);
    }
    if x < 1.0 {
      return Some((-self.lambda).exp());
    }
    Some(gamma_q(x.floor() + 1.0, self.lambda))
  }

  /// The smallest count with `cdf ≥ p`, bracketed from `λ + Φ⁻¹(p)√λ` and bisected over the doubles, so it ends where counts are a double apart; `+∞` past `f64::MAX`.
  fn quantile(&self, p: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) {
      return Some(f64::NAN);
    }
    if p == 0.0 {
      return Some(0.0);
    }
    if p == 1.0 {
      return Some(f64::INFINITY);
    }
    let reaches = |k: f64| self.cdf(k).is_some_and(|c| c >= p);
    let root = self.lambda.sqrt();
    let guess = (self.lambda + ndtri(p) * root).floor().max(0.0);
    let (mut lo, mut hi, mut step) = (guess, guess, root.ceil());
    if reaches(guess) {
      loop {
        lo = (hi - step).min(hi.next_down()).max(-1.0);
        if !reaches(lo) {
          break;
        }
        hi = lo;
        step *= 2.0;
      }
    } else {
      loop {
        hi = (lo + step).min(f64::MAX).max(lo.next_up());
        if reaches(hi) {
          break;
        }
        lo = hi;
        step *= 2.0;
      }
    }
    loop {
      let mid = (lo + (hi - lo) * 0.5).floor();
      if mid <= lo || mid >= hi {
        return Some(hi);
      }
      if reaches(mid) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
  }

  fn mean(&self) -> Option<f64> {
    Some(self.lambda())
  }

  /// Exact from Choi's bracket `λ − ln 2 ≤ m < λ + 1/3`: one count, or two where `frac(λ) ∈ (2/3, ln 2]`, which one cdf comparison decides, except at `n = 0`, where `e^{−λ} ≥ ½` gives 0 without it.
  fn median(&self) -> Option<f64> {
    let n = self.lambda.floor();
    let frac = self.lambda - n;
    // No double lies between `LN_2` and ln 2, and the fused `3 frac − 2` keeps its sign, so both ends are exact.
    let holds_n = frac <= std::f64::consts::LN_2;
    let holds_next = 3.0_f64.mul_add(frac, -2.0) > 0.0;
    Some(if n > 0.0 && holds_n && holds_next {
      if self.cdf(n).is_some_and(|c| c >= 0.5) {
        n
      } else {
        n + 1.0
      }
    } else if holds_n {
      n
    } else {
      n + 1.0
    })
  }

  fn mode(&self) -> Option<f64> {
    Some(self.lambda().floor())
  }

  fn variance(&self) -> Option<f64> {
    Some(self.lambda())
  }

  fn skewness(&self) -> Option<f64> {
    Some(1.0 / self.lambda().sqrt())
  }

  fn kurtosis(&self) -> Option<f64> {
    Some(1.0 / self.lambda())
  }

  /// Exact below the PTRS threshold (the pmf sum to 50σ), the `O(λ⁻⁴)` asymptotic series above it.
  fn entropy(&self) -> Option<f64> {
    let l = self.lambda();
    if l < PTRS_MIN_LAMBDA {
      let (ln_l, end) = (l.ln(), (l + 50.0 * l.sqrt() + 60.0).ceil());
      let (mut h, mut k) = (0.0, 0.0);
      while k <= end {
        let ln_p = ln_pmf(k, l, ln_l);
        h -= ln_p.exp() * ln_p;
        k += 1.0;
      }
      return Some(h);
    }
    Some(
      0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * l).ln()
        - 1.0 / (12.0 * l)
        - 1.0 / (24.0 * l * l)
        - 19.0 / (360.0 * l.powi(3)),
    )
  }

  fn characteristic_function(&self, t: f64) -> Option<num_complex::Complex64> {
    // φ(t) = exp(λ (e^{it} - 1))
    let eit = num_complex::Complex64::new(0.0, t).exp();
    Some(
      (eit - num_complex::Complex64::new(1.0, 0.0))
        .scale(self.lambda())
        .exp(),
    )
  }

  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    Some((self.lambda() * (t.exp() - 1.0)).exp())
  }
}

py_distribution_int!(PyPoissonD, SimdPoisson,
  sig: (lambda_, seed=None),
  params: (lambda_: f64)
);

#[cfg(test)]
mod tests;
