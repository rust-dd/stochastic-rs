use super::super::heston::MultiHestonParams;
use super::payoff::MtPayoff;
use crate::traits::FloatExt;

/// Malliavin–Thalmaier Greeks engine for the supported conditional model.
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
  /// Positive Poisson-kernel regularisation parameter.
  ///
  /// Its bias/variance trade-off is payoff- and model-dependent; the paper
  /// does not prescribe a universal numerical interval.
  pub h: T,
  pub n_paths: usize,
}

impl<T: FloatExt + ndarray_linalg::Lapack> MtGreeks<T> {
  /// Construct an M-T engine for the given model, regularization level and
  /// Monte Carlo path count.
  pub fn new(params: MultiHestonParams<T>, h: T, n_paths: usize) -> Self {
    Self::try_new(params, h, n_paths).expect("invalid Malliavin–Thalmaier engine parameters")
  }

  /// Fallible constructor with model and numerical-parameter validation.
  pub fn try_new(params: MultiHestonParams<T>, h: T, n_paths: usize) -> anyhow::Result<Self> {
    params.validate()?;
    if !num_traits::Float::is_finite(h) || h <= T::zero() {
      anyhow::bail!("h must be finite and positive");
    }
    if n_paths == 0 {
      anyhow::bail!("n_paths must be positive");
    }
    Ok(Self { params, h, n_paths })
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
      let terminal = sample().terminal_prices_array();
      sum += disc
        * payoff.evaluate(
          terminal
            .as_slice()
            .expect("terminal-price array must be contiguous"),
        );
    }
    sum / T::from_usize_(self.n_paths)
  }
}
