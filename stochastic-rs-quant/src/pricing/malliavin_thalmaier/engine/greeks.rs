use super::super::heston::MultiHestonParams;
use super::payoff::MtPayoff;
use crate::traits::FloatExt;

/// Malliavin–Thalmaier Greeks engine.
///
/// Computes multi-asset Greeks using a single Skorohod integral per
/// component, regardless of dimension.
///
/// # Example
///
/// ```ignore
/// let engine = MtGreeks::new(params, 0.01.into(), 50_000);
/// let payoff = MtPayoff::DigitalPut2D { strikes: [100.0, 100.0] };
/// let deltas = engine.all_deltas(&payoff);
/// ```
#[derive(Debug, Clone)]
pub struct MtGreeks<T: FloatExt> {
  pub params: MultiHestonParams<T>,
  /// Regularisation parameter `h`. Recommended `∈ [0.001, 0.1]`.
  pub h: T,
  pub n_paths: usize,
}

impl<T: FloatExt + ndarray_linalg::Lapack> MtGreeks<T> {
  /// Construct an M-T engine for the given model, regularization level and
  /// Monte Carlo path count.
  pub fn new(params: MultiHestonParams<T>, h: T, n_paths: usize) -> Self {
    Self { params, h, n_paths }
  }

  /// Plain Monte Carlo price.
  pub fn price(&self, payoff: &MtPayoff<T>) -> T {
    self.price_from_sampler(payoff, || self.params.sample())
  }

  /// Deterministic variant of [`price`](Self::price).
  pub fn price_with_seed(&self, payoff: &MtPayoff<T>, seed: u64) -> T {
    let mut seed_state = seed;
    self.price_from_sampler(payoff, || {
      self
        .params
        .sample_with_seed(crate::simd_rng::derive_seed(&mut seed_state))
    })
  }

  fn price_from_sampler<F>(&self, payoff: &MtPayoff<T>, mut sample: F) -> T
  where
    F: FnMut() -> super::super::heston::MultiHestonPaths<T>,
  {
    let disc = <T as num_traits::Float>::exp(-self.params.r * self.params.tau);
    let mut sum = T::zero();
    for _ in 0..self.n_paths {
      let st = sample().terminal_prices();
      sum += disc * payoff.evaluate(&st);
    }
    sum / T::from_usize_(self.n_paths)
  }
}
