use super::greeks::MtGreeks;
use super::payoff::MtPayoff;
use crate::traits::FloatExt;

impl<T: FloatExt + ndarray_linalg::Lapack> MtGreeks<T> {
  /// Delta for a given asset via the M-T formula.
  pub fn delta(&self, payoff: &MtPayoff<T>, asset: usize) -> T {
    self.delta_from_sampler(payoff, asset, || self.params.sample())
  }

  /// Deterministic Delta estimator for reproducible tests and benchmarks.
  pub fn delta_with_seed(&self, payoff: &MtPayoff<T>, asset: usize, seed: u64) -> T {
    let mut seed_state = seed;
    self.delta_from_sampler(payoff, asset, || {
      self
        .params
        .sample_with_seed(crate::simd_rng::derive_seed(&mut seed_state))
    })
  }

  fn delta_from_sampler<F>(&self, payoff: &MtPayoff<T>, asset: usize, mut sample: F) -> T
  where
    F: FnMut() -> super::super::heston::MultiHestonPaths<T>,
  {
    let d = self.params.n_assets();
    let discount = <T as num_traits::Float>::exp(-self.params.r * self.params.tau);
    let spots: Vec<T> = self.params.assets.iter().map(|a| a.s0).collect();
    let localization = self.localization(payoff);
    let mut sum = T::zero();

    for _ in 0..self.n_paths {
      let paths = sample();
      let st = paths.terminal_prices();
      let gamma_inv = paths.gamma_inv(&self.params.cross_corr, self.params.tau);
      let h_w = paths.malliavin_weights(&gamma_inv, asset, self.params.r, self.params.tau, &spots);
      let g = match localization.as_ref() {
        Some(loc) => self.compute_g_localized(payoff, &st, loc),
        None => self.compute_g(payoff, &st),
      };
      let pw_contrib = localization
        .as_ref()
        .map(|loc| {
          let grad = self.pathwise_tail_gradient(payoff, &st, loc);
          grad[asset] * st[asset] / spots[asset]
        })
        .unwrap_or_else(T::zero);

      let contrib = (0..d)
        .map(|i| g[[i, asset]] * h_w[i])
        .fold(T::zero(), |a, b| a + b);
      sum += discount * (contrib + pw_contrib);
    }

    sum / T::from_usize_(self.n_paths)
  }

  /// All Deltas in a single simulation pass.
  pub fn all_deltas(&self, payoff: &MtPayoff<T>) -> Vec<T> {
    self.all_deltas_from_sampler(payoff, || self.params.sample())
  }

  /// Deterministic variant of [`all_deltas`](Self::all_deltas).
  pub fn all_deltas_with_seed(&self, payoff: &MtPayoff<T>, seed: u64) -> Vec<T> {
    let mut seed_state = seed;
    self.all_deltas_from_sampler(payoff, || {
      self
        .params
        .sample_with_seed(crate::simd_rng::derive_seed(&mut seed_state))
    })
  }

  fn all_deltas_from_sampler<F>(&self, payoff: &MtPayoff<T>, mut sample: F) -> Vec<T>
  where
    F: FnMut() -> super::super::heston::MultiHestonPaths<T>,
  {
    let d = self.params.n_assets();
    let discount = <T as num_traits::Float>::exp(-self.params.r * self.params.tau);
    let spots: Vec<T> = self.params.assets.iter().map(|a| a.s0).collect();
    let localization = self.localization(payoff);
    let mut sums = vec![T::zero(); d];

    for _ in 0..self.n_paths {
      let paths = sample();
      let st = paths.terminal_prices();
      let gamma_inv = paths.gamma_inv(&self.params.cross_corr, self.params.tau);
      let g = match localization.as_ref() {
        Some(loc) => self.compute_g_localized(payoff, &st, loc),
        None => self.compute_g(payoff, &st),
      };
      let grad_pw = localization
        .as_ref()
        .map(|loc| self.pathwise_tail_gradient(payoff, &st, loc));

      for p in 0..d {
        let h_w = paths.malliavin_weights(&gamma_inv, p, self.params.r, self.params.tau, &spots);
        let contrib = (0..d)
          .map(|i| g[[i, p]] * h_w[i])
          .fold(T::zero(), |a, b| a + b);
        let pw_contrib = grad_pw
          .as_ref()
          .map(|grad| grad[p] * st[p] / spots[p])
          .unwrap_or_else(T::zero);
        sums[p] += discount * (contrib + pw_contrib);
      }
    }

    sums
      .iter()
      .map(|&s| s / T::from_usize_(self.n_paths))
      .collect()
  }
}
