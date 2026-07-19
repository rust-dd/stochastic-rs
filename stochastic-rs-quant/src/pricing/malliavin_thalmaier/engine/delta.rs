use ndarray::Array1;

use super::greeks::MtGreeks;
use super::payoff::MtPayoff;
use crate::traits::FloatExt;

impl<T: FloatExt + ndarray_linalg::Lapack> MtGreeks<T> {
  /// Delta for one asset using exact conditional Malliavin weights.
  ///
  /// This compatibility wrapper panics when the model is outside the supported
  /// conditional-weight scope. Use [`Self::try_delta`] for user-supplied model
  /// parameters.
  pub fn delta(&self, payoff: &MtPayoff<T>, asset: usize) -> T {
    self
      .try_delta(payoff, asset)
      .expect("M-T Delta inputs are unsupported; use try_delta for a recoverable error")
  }

  /// Fallible Delta estimator for the conditional Malliavin-weight scope.
  ///
  /// Stochastic Heston variance is supported when `rho == 0`. Deterministic
  /// variance (`xi == 0`) is supported for any otherwise valid `rho`. A model
  /// with both `xi != 0` and `rho != 0` returns an error because its variance
  /// shock is correlated with the price noise and needs the full Skorohod
  /// derivative of the variance path.
  pub fn try_delta(&self, payoff: &MtPayoff<T>, asset: usize) -> anyhow::Result<T> {
    self.validate_delta_request(payoff, Some(asset))?;
    self.try_delta_from_sampler(payoff, asset, || self.params.sample())
  }

  /// Deterministic Delta estimator for reproducible tests and benchmarks.
  ///
  /// This compatibility wrapper has the same supported scope as [`Self::delta`]
  /// and panics outside it. Use [`Self::try_delta_with_seed`] to handle errors.
  pub fn delta_with_seed(&self, payoff: &MtPayoff<T>, asset: usize, seed: u64) -> T {
    self
      .try_delta_with_seed(payoff, asset, seed)
      .expect("M-T Delta inputs are unsupported; use try_delta_with_seed")
  }

  /// Fallible deterministic variant of [`Self::try_delta`].
  pub fn try_delta_with_seed(
    &self,
    payoff: &MtPayoff<T>,
    asset: usize,
    seed: u64,
  ) -> anyhow::Result<T> {
    self.validate_delta_request(payoff, Some(asset))?;
    let mut seed_state = seed;
    self.try_delta_from_sampler(payoff, asset, || {
      self
        .params
        .sample_with_seed(crate::simd_rng::derive_seed(&mut seed_state))
    })
  }

  fn try_delta_from_sampler<F>(
    &self,
    payoff: &MtPayoff<T>,
    asset: usize,
    mut sample: F,
  ) -> anyhow::Result<T>
  where
    F: FnMut() -> super::super::heston::MultiHestonPaths<T>,
  {
    let d = self.params.n_assets();
    let discount = <T as num_traits::Float>::exp(-self.params.r * self.params.tau);
    let spots = Array1::from_iter(self.params.assets.iter().map(|params| params.s0));
    let localization = self.localization(payoff);
    let mut sum = T::zero();

    for _ in 0..self.n_paths {
      let paths = sample();
      let terminal = paths.terminal_prices_array();
      let terminal_slice = terminal
        .as_slice()
        .expect("terminal-price array must be contiguous");
      let gamma_inv = paths.try_gamma_inv(&self.params.cross_corr, self.params.tau)?;
      let weights =
        paths.try_malliavin_weights(&gamma_inv, asset, self.params.r, self.params.tau, &spots)?;
      let g = match localization.as_ref() {
        Some(loc) => self.compute_g_localized(payoff, terminal_slice, loc),
        None => self.compute_g(payoff, terminal_slice),
      };
      let pathwise = localization
        .as_ref()
        .map(|loc| {
          let gradient = self.pathwise_tail_gradient(payoff, terminal_slice, loc);
          gradient[asset] * terminal[asset] / spots[asset]
        })
        .unwrap_or_else(T::zero);
      let mt_contribution = (0..d)
        .map(|i| g[[i, asset]] * weights[i])
        .fold(T::zero(), |accumulator, value| accumulator + value);
      sum += discount * (mt_contribution + pathwise);
    }

    Ok(sum / T::from_usize_(self.n_paths))
  }

  /// All Deltas in a single simulation pass.
  ///
  /// This compatibility wrapper panics outside the exact conditional-weight
  /// scope. Use [`Self::try_all_deltas`] for a fallible result backed by an
  /// [`Array1`].
  pub fn all_deltas(&self, payoff: &MtPayoff<T>) -> Vec<T> {
    self
      .try_all_deltas(payoff)
      .expect("M-T Delta inputs are unsupported; use try_all_deltas")
      .to_vec()
  }

  /// Fallible single-pass estimator of every Delta.
  pub fn try_all_deltas(&self, payoff: &MtPayoff<T>) -> anyhow::Result<Array1<T>> {
    self.validate_delta_request(payoff, None)?;
    self.try_all_deltas_from_sampler(payoff, || self.params.sample())
  }

  /// Deterministic variant of [`Self::all_deltas`].
  ///
  /// This compatibility wrapper panics outside the supported scope. Use
  /// [`Self::try_all_deltas_with_seed`] for a fallible result.
  pub fn all_deltas_with_seed(&self, payoff: &MtPayoff<T>, seed: u64) -> Vec<T> {
    self
      .try_all_deltas_with_seed(payoff, seed)
      .expect("M-T Delta inputs are unsupported; use try_all_deltas_with_seed")
      .to_vec()
  }

  /// Fallible deterministic variant of [`Self::try_all_deltas`].
  pub fn try_all_deltas_with_seed(
    &self,
    payoff: &MtPayoff<T>,
    seed: u64,
  ) -> anyhow::Result<Array1<T>> {
    self.validate_delta_request(payoff, None)?;
    let mut seed_state = seed;
    self.try_all_deltas_from_sampler(payoff, || {
      self
        .params
        .sample_with_seed(crate::simd_rng::derive_seed(&mut seed_state))
    })
  }

  fn try_all_deltas_from_sampler<F>(
    &self,
    payoff: &MtPayoff<T>,
    mut sample: F,
  ) -> anyhow::Result<Array1<T>>
  where
    F: FnMut() -> super::super::heston::MultiHestonPaths<T>,
  {
    let d = self.params.n_assets();
    let discount = <T as num_traits::Float>::exp(-self.params.r * self.params.tau);
    let spots = Array1::from_iter(self.params.assets.iter().map(|params| params.s0));
    let localization = self.localization(payoff);
    let mut sums = Array1::<T>::zeros(d);

    for _ in 0..self.n_paths {
      let paths = sample();
      let terminal = paths.terminal_prices_array();
      let terminal_slice = terminal
        .as_slice()
        .expect("terminal-price array must be contiguous");
      let gamma_inv = paths.try_gamma_inv(&self.params.cross_corr, self.params.tau)?;
      let g = match localization.as_ref() {
        Some(loc) => self.compute_g_localized(payoff, terminal_slice, loc),
        None => self.compute_g(payoff, terminal_slice),
      };
      let pathwise_gradient = localization
        .as_ref()
        .map(|loc| self.pathwise_tail_gradient(payoff, terminal_slice, loc));

      for asset in 0..d {
        let weights =
          paths.try_malliavin_weights(&gamma_inv, asset, self.params.r, self.params.tau, &spots)?;
        let mt_contribution = (0..d)
          .map(|i| g[[i, asset]] * weights[i])
          .fold(T::zero(), |accumulator, value| accumulator + value);
        let pathwise = pathwise_gradient
          .as_ref()
          .map(|gradient| gradient[asset] * terminal[asset] / spots[asset])
          .unwrap_or_else(T::zero);
        sums[asset] += discount * (mt_contribution + pathwise);
      }
    }

    Ok(sums.mapv(|sum| sum / T::from_usize_(self.n_paths)))
  }

  fn validate_delta_request(
    &self,
    payoff: &MtPayoff<T>,
    requested_asset: Option<usize>,
  ) -> anyhow::Result<()> {
    self.params.validate()?;
    self.params.validate_conditional_malliavin_weights()?;
    let d = self.params.n_assets();
    if d < 2 {
      anyhow::bail!("Malliavin-Thalmaier Delta requires at least two assets, got {d}");
    }
    if self.n_paths == 0 {
      anyhow::bail!("n_paths must be positive");
    }
    if !self.h.is_finite() || self.h <= T::zero() {
      anyhow::bail!(
        "regularization h must be finite and positive, got {:?}",
        self.h
      );
    }
    if !self.params.tau.is_finite() || self.params.tau <= T::zero() {
      anyhow::bail!(
        "maturity tau must be finite and positive, got {:?}",
        self.params.tau
      );
    }
    if let Some(asset) = requested_asset
      && asset >= d
    {
      anyhow::bail!("Delta asset={asset} is out of bounds for n_assets={d}");
    }
    for (asset, params) in self.params.assets.iter().enumerate() {
      if !params.s0.is_finite() || params.s0 <= T::zero() {
        anyhow::bail!("initial spot for asset {asset} must be finite and positive");
      }
    }
    match payoff {
      MtPayoff::Call { asset, strike } | MtPayoff::Put { asset, strike } => {
        if *asset >= d {
          anyhow::bail!("payoff asset={asset} is out of bounds for n_assets={d}");
        }
        if !strike.is_finite() {
          anyhow::bail!("payoff strike must be finite");
        }
      }
      MtPayoff::DigitalPut2D { strikes } => {
        if d != 2 {
          anyhow::bail!("DigitalPut2D requires exactly two assets, got {d}");
        }
        if strikes
          .iter()
          .any(|strike| !strike.is_finite() || *strike <= T::zero())
        {
          anyhow::bail!("DigitalPut2D strikes must be finite and positive");
        }
      }
      MtPayoff::BasketCall { weights, strike } => {
        if weights.len() != d {
          anyhow::bail!(
            "basket weights length {} does not match n_assets={d}",
            weights.len()
          );
        }
        if weights.iter().any(|weight| !weight.is_finite()) {
          anyhow::bail!("all basket weights must be finite");
        }
        if !strike.is_finite() {
          anyhow::bail!("payoff strike must be finite");
        }
      }
      MtPayoff::WorstOfPut { strike } => {
        if !strike.is_finite() {
          anyhow::bail!("payoff strike must be finite");
        }
      }
    }
    Ok(())
  }
}

#[cfg(test)]
#[path = "delta_tests.rs"]
mod delta_tests;
