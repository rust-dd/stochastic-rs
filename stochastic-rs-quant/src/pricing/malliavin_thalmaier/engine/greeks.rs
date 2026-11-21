use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::super::heston::MultiHestonParams;
use super::super::heston::MultiHestonPaths;
use super::payoff::MtPayoff;
use crate::QuantError;
use crate::traits::FloatExt;

/// Malliavin–Thalmaier Greeks engine for the supported conditional model.
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use stochastic_rs_core::simd_rng::Deterministic;
/// use stochastic_rs_quant::pricing::malliavin_thalmaier::{
///     AssetParams, MtGreeks, MtPayoff, MultiHestonParams,
/// };
///
/// let first = AssetParams { s: 100.0, v0: 0.0625, kappa: 1.0, theta: 0.0625, xi: 0.0, rho: 0.0 };
/// let second = AssetParams { s: 100.0, v0: 0.04, kappa: 1.0, theta: 0.04, xi: 0.0, rho: 0.0 };
/// let mut cross_corr = Array2::<f64>::eye(2);
/// cross_corr[[0, 1]] = 0.2;
/// cross_corr[[1, 0]] = 0.2;
/// let params = MultiHestonParams {
///     assets: vec![first, second], cross_corr, r: 0.0, tau: 1.0, n_steps: 9,
/// };
///
/// let engine = MtGreeks::new(params, 0.01, 16).with_seed(Deterministic::new(42));
/// let payoff = MtPayoff::DigitalPut2D { strikes: [100.0, 100.0] };
/// let deltas = engine.all_deltas(&payoff);
/// assert_eq!(deltas.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct MtGreeks<T: FloatExt, S: SeedExt = Unseeded> {
  pub(super) params: MultiHestonParams<T>,
  /// Positive Poisson-kernel regularisation `h`; the bias/variance trade-off depends on payoff and
  /// model, and the paper prescribes no interval (the true Greek is the `h → 0` limit).
  pub(super) h: T,
  pub(super) n_paths: usize,
  /// Each estimate draws its paths' seeds from a clone, and a bump pair reads one draw of them,
  /// its common random numbers.
  pub seed: S,
}

impl<T: FloatExt> MtGreeks<T> {
  /// Construct an M-T engine for the given model, regularization level and
  /// Monte Carlo path count.
  pub fn new(params: MultiHestonParams<T>, h: T, n_paths: usize) -> Self {
    Self::try_new(params, h, n_paths).unwrap_or_else(|e| panic!("{e}"))
  }

  /// Fallible constructor with model and numerical-parameter validation.
  pub fn try_new(params: MultiHestonParams<T>, h: T, n_paths: usize) -> Result<Self, QuantError> {
    params.validate()?;
    if !num_traits::Float::is_finite(h) || h <= T::zero() {
      return Err(QuantError::invalid(
        "h",
        format!("must be finite and positive, got {h:?}"),
      ));
    }
    if n_paths == 0 {
      return Err(QuantError::invalid("n_paths", "must be positive, got 0"));
    }
    Ok(Self {
      params,
      h,
      n_paths,
      seed: Unseeded,
    })
  }
}

impl<T: FloatExt, S: SeedExt> MtGreeks<T, S> {
  pub fn with_seed<S2: SeedExt>(self, seed: S2) -> MtGreeks<T, S2> {
    MtGreeks {
      params: self.params,
      h: self.h,
      n_paths: self.n_paths,
      seed,
    }
  }

  /// Plain Monte Carlo price.
  pub fn price(&self, payoff: &MtPayoff<T>) -> T {
    self.price_at(payoff, &self.path_seeds())
  }

  pub(super) fn path_seeds(&self) -> Vec<u64> {
    let seed = self.seed.clone();
    (0..self.n_paths).map(|_| seed.next_seed()).collect()
  }

  pub(super) fn sampler<'a>(
    &'a self,
    seeds: &'a [u64],
  ) -> impl FnMut() -> MultiHestonPaths<T> + 'a {
    let mut seeds = seeds.iter();
    move || {
      let seed = seeds.next().expect("one seed per path");
      self.params.sample(&Deterministic::new(*seed))
    }
  }

  pub(super) fn price_at(&self, payoff: &MtPayoff<T>, seeds: &[u64]) -> T {
    self.price_from_sampler(payoff, self.sampler(seeds))
  }

  fn price_from_sampler<F>(&self, payoff: &MtPayoff<T>, mut sample: F) -> T
  where
    F: FnMut() -> MultiHestonPaths<T>,
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
