use super::greeks::MtGreeks;
use super::payoff::MtPayoff;
use crate::traits::FloatExt;

impl<T: FloatExt + ndarray_linalg::Lapack> MtGreeks<T> {
  /// Cross-Gamma via finite difference on Delta.
  pub fn cross_gamma(&self, payoff: &MtPayoff<T>, asset_a: usize, asset_b: usize) -> T {
    let bump = self.params.assets[asset_b].s0 * T::from_f64_fast(0.01);
    let mut up = self.params.clone();
    up.assets[asset_b].s0 += bump;
    let mut dn = self.params.clone();
    dn.assets[asset_b].s0 -= bump;

    let d_up = MtGreeks::new(up, self.h, self.n_paths).delta(payoff, asset_a);
    let d_dn = MtGreeks::new(dn, self.h, self.n_paths).delta(payoff, asset_a);
    (d_up - d_dn) / (bump + bump)
  }

  /// Backward-compatible alias retained for the integration tests.
  pub fn cross_gamma_fd(&self, payoff: &MtPayoff<T>, asset_a: usize, asset_b: usize) -> T {
    self.cross_gamma(payoff, asset_a, asset_b)
  }

  /// Vega via finite difference on price.
  ///
  /// Uses a fixed-seed common-random-numbers bump internally to reduce Monte
  /// Carlo variance compared with two independent price runs.
  pub fn vega(&self, payoff: &MtPayoff<T>, asset: usize) -> T {
    self.vega_with_seed(payoff, asset, 0x9E37_79B9_7F4A_7C15_u64 ^ asset as u64)
  }

  /// Deterministic Vega estimator using common random numbers for the
  /// up/down finite-difference bump.
  pub fn vega_with_seed(&self, payoff: &MtPayoff<T>, asset: usize, seed: u64) -> T {
    let bump = self.params.assets[asset].v0 * T::from_f64_fast(0.01);
    let mut up = self.params.clone();
    up.assets[asset].v0 += bump;
    let mut dn = self.params.clone();
    dn.assets[asset].v0 -= bump;

    let p_up = MtGreeks::new(up, self.h, self.n_paths).price_with_seed(payoff, seed);
    let p_dn = MtGreeks::new(dn, self.h, self.n_paths).price_with_seed(payoff, seed);
    (p_up - p_dn) / (bump + bump)
  }
}
