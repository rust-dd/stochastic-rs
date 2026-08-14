//! # stochastic-rs-distributions
//!
//! Probability distributions with SIMD bulk sampling, plus the foundational
//! `FloatExt` / `SimdFloatExt` trait machinery and float impls.
//!
//! ## Choosing a distribution
//!
//! The website's distributions catalog groups these 29 types (30 counting
//! [`complex::ComplexDistribution`]) by sampling strategy; this section
//! groups them by modeling role instead — the axis that actually decides
//! which one to reach for.
//!
//! **Unbounded, light-tailed**: [`normal::SimdNormal`] is the default; the
//! only other unbounded-symmetric type is [`ged::SimdGed`], whose shape
//! parameter `β` tunes peakedness (`β=1` Laplace, `β=2` Gaussian, `β<2`
//! more peaked, `β>2` flatter) but — despite "generalized error
//! distribution" sounding fat-tailed — its density decays as
//! $e^{-|z|^\beta}$ for *any* `β > 0`, which is always lighter than a
//! power law. Reach for [`studentt::SimdStudentT`] or
//! [`pareto::SimdPareto`] below, not a small `β`, when you actually need
//! polynomial tail decay.
//!
//! **Unbounded, heavy-tailed** (financial returns, extreme risk):
//! [`cauchy::SimdCauchy`] (no mean or variance, ever) and
//! [`studentt::SimdStudentT`] (moment-existence thresholds keyed to `ν`:
//! mean undefined for `ν ≤ 1`, variance infinite for `ν ≤ 2`) are the same
//! distribution at one point — `StudentT(ν=1)` **is** Cauchy exactly, not
//! approximately. [`alpha_stable::SimdAlphaStable`] generalizes further
//! (`α=2` is [`normal::SimdNormal`], `α=1, β=0` is
//! [`cauchy::SimdCauchy`], again exactly), but at general `α` it has no
//! closed-form `pdf`/`cdf`/`inv_cdf` at all and often infinite variance —
//! reach for it only when the stable/self-similar-sum property itself
//! matters, not merely "heavy tails." [`normal_inverse_gauss::SimdNormalInverseGauss`]
//! is the usual alternative when you want a *skewed* heavy tail with
//! **finite** variance and full closed-form moments (Barndorff-Nielsen
//! 1997, *Scandinavian Journal of Statistics* 24(1), 1-13, DOI:
//! 10.1111/1467-9469.00045) — the practical reason to prefer NIG over
//! [`alpha_stable::SimdAlphaStable`] for return modeling specifically.
//! [`gev::SimdGev`] is a different job again: block-maxima / extreme-value
//! modeling, not the bulk return distribution.
//!
//! **Positive support** (durations, volatilities, waiting times):
//! [`exp::SimdExp`] has constant hazard (memoryless); [`weibull::SimdWeibull`]
//! generalizes it with a hazard that can rise or fall over time, and is
//! built directly on [`exp::SimdExpZig`] — the same ziggurat primitive
//! [`exp::SimdExp`] itself wraps.
//! [`gamma::SimdGamma`] is the waiting-time-for-`k`-events story and the
//! internal building block for [`beta::SimdBeta`] (ratio of two Gammas),
//! [`chi_square::SimdChiSquared`] (`= 2·Gamma(k/2, 1)`),
//! [`ged::SimdGed`] and [`dirichlet::SimdDirichlet`].
//! [`lognormal::SimdLogNormal`] is the multiplicative-growth story (e.g.
//! GBM levels); [`inverse_gauss::SimdInverseGauss`] is a first-passage-time
//! distribution and the subordinator inside
//! [`normal_inverse_gauss::SimdNormalInverseGauss`]. Only
//! [`pareto::SimdPareto`] among this group is genuinely power-law
//! heavy-tailed, with explicit threshold behavior (mean finite only for
//! `α > 1`, variance only for `α > 2` — the crate's own test fixes
//! `α = 1.16`, the classic "80/20" case, as finite-mean/infinite-variance).
//!
//! **Bounded support**: [`beta::SimdBeta`] and [`uniform::SimdUniform`]
//! are the two general-purpose bounded distributions.
//! [`truncated::SimdTruncatedNormal`], [`truncated::SimdTruncatedExp`],
//! [`truncated::SimdTruncatedBeta`] and [`truncated::SimdTruncatedGamma`]
//! are not a family to choose among — each restricts one specific base
//! distribution to an interval, so the choice is just "which base did you
//! already want." Two of the four have a real failure mode worth knowing
//! before you truncate to a very tight interval: when rejection
//! acceptance falls below the crate's threshold,
//! [`truncated::SimdTruncatedBeta`] and [`truncated::SimdTruncatedGamma`]
//! silently return the clamped interval midpoint rather than a genuine
//! draw — widen the interval if you see draws piling up on one value.
//! [`truncated::SimdTruncatedNormal`] and [`truncated::SimdTruncatedExp`]
//! have no such fallback: tight-interval Normal routes through an exact
//! closed-form inverse-CDF instead, and Exponential never needed rejection
//! in the first place.
//!
//! **Discrete counts**: [`poisson::SimdPoisson`] (constant-rate count) and
//! [`binomial::SimdBinomial`] (fixed trials, *with* replacement) are the
//! two defaults. [`hypergeometric::SimdHypergeometric`] is the one to use
//! instead of Binomial when sampling is *without* replacement from a
//! finite population — that assumption, not a tuning knob, is what
//! separates them. [`geometric::SimdGeometric`] is the discrete analogue
//! of [`exp::SimdExp`]'s memorylessness (waiting time to first success).
//! [`skellam::SimdSkellam`] — the difference of two independent
//! [`poisson::SimdPoisson`] draws — is the only discrete type here that
//! can go negative, useful for signed count data such as net order-flow
//! imbalance.
//!
//! **Multivariate and structural**: [`dirichlet::SimdDirichlet`] (simplex-
//! valued, conjugate prior for Categorical/Multinomial proportions) and
//! [`wishart::SimdWishart`] (SPD-matrix-valued, Bartlett decomposition —
//! the standard way to generate random covariance/correlation matrices)
//! both skip [`crate::traits::DistributionExt`] entirely in favor of their
//! own inherent `pdf`/`log_pdf` methods, since that trait's `f64 -> f64`
//! shape does not fit a vector- or matrix-valued draw.
//! [`non_central_chi_squared::SimdNonCentralChiSquared`] is structurally
//! different again: its noncentrality parameter is supplied per draw
//! (`sample_ncp(ncp)`, not at construction), it implements neither
//! `rand_distr::Distribution` nor [`crate::traits::DistributionExt`], and
//! its purpose in this crate is narrow — backing the exact CIR transition
//! density (`stochastic-rs-stats::cir`). For `0 < df < 1`, where the
//! Gaussian-shift decomposition doesn't exist, `sample_ncp` falls back per
//! draw to the same Poisson-mixture branch the free function
//! [`non_central_chi_squared::sample`] uses; buffering only benefits the
//! `df ≥ 1` path, so prefer the free function for isolated `df < 1` draws
//! outside a per-step loop.
//! [`complex::ComplexDistribution`] is a composition utility, not a shape
//! family — it pairs two independent distributions as the real and
//! imaginary parts of a `Complex<T>` draw.
//!
//! ## `DistributionExt` coverage is uneven — check before you rely on it
//!
//! A type "implementing [`crate::traits::DistributionExt`]" does not mean
//! every method has a closed form. [`ged::SimdGed`], [`gev::SimdGev`] and
//! [`skellam::SimdSkellam`] each override only `pdf`/`cdf`; the four
//! [`truncated`] wrappers do the same. None of these seven expose
//! `mean`/`variance`/`inv_cdf`/etc. — calling them panics with the
//! default `unimplemented!("... not implemented for {type_name}")`, even
//! where the closed form is textbook (`SimdGev`'s own test file computes
//! its mean via `Γ(1-ξ)` inline, but that formula is not exposed as
//! `.mean()`). [`dirichlet::SimdDirichlet`], [`wishart::SimdWishart`],
//! [`non_central_chi_squared::SimdNonCentralChiSquared`] and
//! [`complex::ComplexDistribution`] implement none of it at all — see the
//! cluster notes above for what each offers instead. If you need a
//! specific moment programmatically rather than deriving it by hand,
//! confirm the override exists (`cargo doc -p stochastic-rs-distributions`
//! renders exactly what's overridden) before depending on it.
//!
//! One further, unrelated note while it's on this page:
//! [`geometric::SimdGeometric`]'s documented domain for its success
//! probability is `p ∈ (0, 1]`; `SimdGeometric::new` asserts it, the same
//! way [`binomial::SimdBinomial::new`] asserts its own `p ∈ [0, 1]` — the
//! closed upper endpoint `p = 1.0` (the entropy override's own degenerate
//! case) stays valid rather than being excluded.

// Defaults to `warn`, which is how 5 broken doc links accumulated
// unnoticed; deny so a regression fails the build instead of drifting.
#![deny(rustdoc::broken_intra_doc_links)]
#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

pub use stochastic_rs_core::simd_rng;

#[macro_use]
mod macros;

pub mod float_impls;
mod simd_float_impls;
pub mod special;
pub mod traits;

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
pub use crate::traits::CallableDist;
pub use crate::traits::DistributionExt;
pub use crate::traits::DistributionSampler;
pub use crate::traits::FloatExt;
pub use crate::traits::Fn1D;
pub use crate::traits::Fn2D;
pub use crate::traits::SimdFloatExt;

pub mod alpha_stable;
pub mod beta;
pub mod binomial;
pub mod cauchy;
pub mod chi_square;
pub mod complex;
pub mod exp;
pub mod gamma;
pub mod geometric;
pub mod hypergeometric;
pub mod inverse_gauss;
pub mod lognormal;
pub mod non_central_chi_squared;
pub mod normal;
pub mod normal_inverse_gauss;

/// Type alias for `SimdNormal` backed by the experimental dual-stream RNG.
///
/// Enabled by the `dual-stream-rng` cargo feature. Production code continues
/// to use the default [`normal::SimdNormal`] alias parameter
/// (`R = SimdRng`); switching to this alias picks the same struct
/// monomorphised over `SimdRngDual`, which unrolls the Ziggurat hot loop
/// 2× for ≈ 5–11 % extra throughput on bulk Normal fills.
#[cfg(feature = "dual-stream-rng")]
pub type SimdNormalDual<T, const N: usize = 64> =
  normal::SimdNormal<T, N, stochastic_rs_core::simd_rng_dual::SimdRngDual>;

/// Type alias for [`exp::SimdExp`] backed by the experimental dual-stream
/// RNG. Same trade-offs as [`SimdNormalDual`].
#[cfg(feature = "dual-stream-rng")]
pub type SimdExpDual<T> = exp::SimdExp<T, stochastic_rs_core::simd_rng_dual::SimdRngDual>;

/// Type alias for [`exp::SimdExpZig`] (the bulk-fill primitive that powers
/// [`exp::SimdExp`]) backed by the dual-stream RNG.
#[cfg(feature = "dual-stream-rng")]
pub type SimdExpZigDual<T, const N: usize = 64> =
  exp::SimdExpZig<T, N, stochastic_rs_core::simd_rng_dual::SimdRngDual>;
pub mod dirichlet;
pub mod ged;
pub mod gev;
pub mod pareto;
pub mod poisson;
pub mod scalar;
pub mod skellam;
pub mod studentt;
pub mod truncated;
pub mod uniform;
pub mod weibull;
pub mod wishart;

macro_rules! impl_distribution_sampler_float {
  ($($dist:ty),+ $(,)?) => {
    $(
      impl<T: SimdFloatExt> DistributionSampler<T> for $dist {
        #[inline]
        fn fill_slice(&self, out: &mut [T]) {
          self.fill_slice(out);
        }

        #[inline]
        fn fork(&self, stream_idx: u64) -> Self {
          self.fork(stream_idx)
        }
      }
    )+
  };
}

macro_rules! impl_distribution_sampler_int {
  ($($dist:ty),+ $(,)?) => {
    $(
      impl<T: num_traits::PrimInt> DistributionSampler<T> for $dist {
        #[inline]
        fn fill_slice(&self, out: &mut [T]) {
          self.fill_slice(out);
        }

        #[inline]
        fn fork(&self, stream_idx: u64) -> Self {
          self.fork(stream_idx)
        }
      }
    )+
  };
}

macro_rules! impl_distribution_sampler_float_const_n {
  ($($dist:ty),+ $(,)?) => {
    $(
      impl<T: SimdFloatExt, const N: usize> DistributionSampler<T> for $dist {
        #[inline]
        fn fill_slice(&self, out: &mut [T]) {
          self.fill_slice(out);
        }

        #[inline]
        fn fork(&self, stream_idx: u64) -> Self {
          self.fork(stream_idx)
        }
      }
    )+
  };
}

impl_distribution_sampler_float!(
  alpha_stable::SimdAlphaStable<T>,
  beta::SimdBeta<T>,
  cauchy::SimdCauchy<T>,
  chi_square::SimdChiSquared<T>,
  exp::SimdExp<T>,
  gamma::SimdGamma<T>,
  ged::SimdGed<T>,
  gev::SimdGev<T>,
  inverse_gauss::SimdInverseGauss<T>,
  lognormal::SimdLogNormal<T>,
  normal_inverse_gauss::SimdNormalInverseGauss<T>,
  pareto::SimdPareto<T>,
  studentt::SimdStudentT<T>,
  uniform::SimdUniform<T>,
  weibull::SimdWeibull<T>,
);

impl_distribution_sampler_int!(
  binomial::SimdBinomial<T>,
  geometric::SimdGeometric<T>,
  hypergeometric::SimdHypergeometric<T>,
  poisson::SimdPoisson<T>,
);

impl_distribution_sampler_float_const_n!(normal::SimdNormal<T, N>, exp::SimdExpZig<T, N>,);

#[cfg(test)]
mod distribution_sampler_tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::SimdRng;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::DistributionSampler;
  use super::normal::SimdNormal;
  use super::poisson::SimdPoisson;

  #[test]
  fn sample_n_returns_requested_length() {
    let dist = SimdNormal::<f64>::new(0.0, 1.0, &Unseeded);
    let out = dist.sample_n(1024);
    assert_eq!(out.len(), 1024);
  }

  #[test]
  fn sample_matrix_float_has_expected_shape() {
    let dist = SimdNormal::<f32>::new(0.0, 1.0, &Unseeded);
    let out = dist.sample_matrix(32, 64);
    assert_eq!(out.shape(), &[32, 64]);
  }

  #[test]
  fn sample_matrix_int_has_expected_shape() {
    let dist = SimdPoisson::<i64>::new(1.5, &Unseeded);
    let out = dist.sample_matrix(16, 8);
    assert_eq!(out.shape(), &[16, 8]);
  }

  /// Two identically-seeded objects must produce identical output through
  /// EVERY public sampling path. Guards the Clone-reseed and fresh-SimdRng
  /// leaks found by the 2026-08-11 API review (empirically: Poisson
  /// `sample_n` and all types' parallel `sample_matrix` diverged under
  /// `Deterministic` seeds because `sample_n` handed a freshly-seeded
  /// `SimdRng` into `fill_slice`, and `SimdPoisson` — unlike most other
  /// types — actually drew from that argument instead of its own stream).
  #[test]
  fn sample_n_deterministic_all_paths() {
    let poisson_a = SimdPoisson::<u64>::new(4.5, &Deterministic::new(42));
    let poisson_b = SimdPoisson::<u64>::new(4.5, &Deterministic::new(42));
    assert_eq!(poisson_a.sample_n(64), poisson_b.sample_n(64));

    let normal_a = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(42));
    let normal_b = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(42));
    assert_eq!(normal_a.sample_n(64), normal_b.sample_n(64));
  }

  /// `sample_matrix`'s parallel fan-out (`rayon::scope` + per-chunk workers)
  /// must stay bit-identical across two identically-`Deterministic`-seeded
  /// samplers. Before the `DistributionSampler::fork` fix, each worker
  /// cloned `self` and every `Simd*` `Clone` impl reseeds from `Unseeded`,
  /// so the seeded stream was lost the moment `sample_matrix` went
  /// multi-threaded. Forced onto a 4-thread pool so the parallel branch
  /// (`workers > 1`) is actually exercised regardless of the ambient
  /// environment's core count.
  #[test]
  fn sample_matrix_parallel_deterministic() {
    let pool = rayon::ThreadPoolBuilder::new()
      .num_threads(4)
      .build()
      .expect("failed to build 4-thread pool");
    let (a, b) = pool.install(|| {
      let dist_a = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(42));
      let dist_b = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(42));
      (
        dist_a.sample_matrix(200, 2000),
        dist_b.sample_matrix(200, 2000),
      )
    });
    assert_eq!(a, b);
  }

  /// A second follow-up review (2026-08-11) caught that the fix above only
  /// proved two *separately-constructed* samplers agree on their *first*
  /// call — `fork` derived every worker's seed from a value frozen at
  /// construction, so a single object's `sample_matrix` replayed the exact
  /// same matrix on every call. "Construct once, call `sample_matrix` per
  /// Monte Carlo iteration" is the library's central usage pattern, so this
  /// was a real bug, not a theoretical one. `fork` now advances an
  /// interior-mutable basis on every parallel-path call; this test asserts
  /// two consecutive calls diverge, for both `Deterministic` and
  /// `Unseeded` (repeat calls must advance regardless of seeding strategy).
  #[test]
  fn sample_matrix_repeat_calls_advance() {
    let pool = rayon::ThreadPoolBuilder::new()
      .num_threads(4)
      .build()
      .expect("failed to build 4-thread pool");
    pool.install(|| {
      let det = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(42));
      let call1 = det.sample_matrix(200, 2000);
      let call2 = det.sample_matrix(200, 2000);
      assert_ne!(
        call1, call2,
        "Deterministic sample_matrix replayed the same matrix on a repeat call"
      );

      let unseeded = SimdNormal::<f64>::new(0.0, 1.0, &Unseeded);
      let call1 = unseeded.sample_matrix(200, 2000);
      let call2 = unseeded.sample_matrix(200, 2000);
      assert_ne!(
        call1, call2,
        "Unseeded sample_matrix replayed the same matrix on a repeat call"
      );
    });
  }

  /// Companion to [`sample_matrix_repeat_calls_advance`]: advancing the
  /// fork basis per call must not break cross-object reproducibility.
  /// Two identically-`Deterministic`-seeded objects still need to agree
  /// call-for-call (their live states advance in lockstep from an
  /// identical starting point), and a small *serial* call sandwiched
  /// between the two parallel calls (below the fork threshold, so it never
  /// touches the fork basis) must not desynchronize them.
  #[test]
  fn sample_matrix_call_sequence_deterministic() {
    let pool = rayon::ThreadPoolBuilder::new()
      .num_threads(4)
      .build()
      .expect("failed to build 4-thread pool");
    pool.install(|| {
      let dist_a = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(42));
      let dist_b = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(42));

      let a_call1 = dist_a.sample_matrix(200, 2000);
      let b_call1 = dist_b.sample_matrix(200, 2000);
      assert_eq!(a_call1, b_call1, "first parallel call diverged");

      // Below MIN_PAR_CHUNK: takes the serial `fill_slice` path, which
      // must not perturb either object's fork basis.
      let a_serial = dist_a.sample_matrix(2, 8);
      let b_serial = dist_b.sample_matrix(2, 8);
      assert_eq!(a_serial, b_serial, "serial call diverged");

      let a_call2 = dist_a.sample_matrix(200, 2000);
      let b_call2 = dist_b.sample_matrix(200, 2000);
      assert_eq!(a_call2, b_call2, "second parallel call diverged");
      assert_ne!(a_call1, a_call2, "second parallel call replayed the first");
    });
  }

  /// sample_matrix must be bit-identical across thread-pool sizes, not merely for
  /// a fixed pool — the A1-a fix left worker count tied to current_num_threads().
  #[test]
  fn sample_matrix_is_thread_count_independent() {
    let sample_under = |threads: usize| {
      rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build pool")
        .install(|| {
          let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(42));
          dist.sample_matrix(200, 2000)
        })
    };
    let under_1 = sample_under(1);
    let under_4 = sample_under(4);
    let under_8 = sample_under(8);
    assert_eq!(under_1, under_4, "1-thread and 4-thread pools diverged");
    assert_eq!(under_4, under_8, "4-thread and 8-thread pools diverged");
  }

  /// `rand_distr::Distribution::sample`'s `rng` argument is documented as
  /// unused across every `Simd*` type — feeding it two genuinely different
  /// external RNGs must not change the output of a `Deterministic`-seeded
  /// sampler.
  #[test]
  fn rand_distr_sample_uses_internal_stream() {
    use rand_distr::Distribution;

    let poisson_a = SimdPoisson::<u64>::new(4.5, &Deterministic::new(7));
    let poisson_b = SimdPoisson::<u64>::new(4.5, &Deterministic::new(7));
    let mut external_rng_1 = SimdRng::from_seed(1);
    let mut external_rng_2 = SimdRng::from_seed(999_999);
    for _ in 0..32 {
      let a = poisson_a.sample(&mut external_rng_1);
      let b = poisson_b.sample(&mut external_rng_2);
      assert_eq!(a, b);
    }
  }
}
