use super::greeks::MtGreeks;
use super::payoff::MtPayoff;
use crate::traits::FloatExt;

impl<T: FloatExt + ndarray_linalg::Lapack> MtGreeks<T> {
  /// Cross-Gamma via a common-random-numbers finite difference on Delta.
  ///
  /// This is a numerical utility, not a higher-order Malliavin--Thalmaier
  /// weight estimator.
  pub fn cross_gamma(&self, payoff: &MtPayoff<T>, asset_a: usize, asset_b: usize) -> T {
    self
      .try_cross_gamma(payoff, asset_a, asset_b)
      .expect("cross-Gamma inputs are unsupported; use try_cross_gamma")
  }

  /// Fallible common-random-numbers cross-Gamma.
  pub fn try_cross_gamma(
    &self,
    payoff: &MtPayoff<T>,
    asset_a: usize,
    asset_b: usize,
  ) -> anyhow::Result<T> {
    self.try_cross_gamma_with_seed(
      payoff,
      asset_a,
      asset_b,
      0xD1B5_4A32_D192_ED03_u64 ^ asset_a as u64 ^ ((asset_b as u64) << 32),
    )
  }

  /// Deterministic cross-Gamma using the same normal draws for both bumps.
  pub fn cross_gamma_with_seed(
    &self,
    payoff: &MtPayoff<T>,
    asset_a: usize,
    asset_b: usize,
    seed: u64,
  ) -> T {
    self
      .try_cross_gamma_with_seed(payoff, asset_a, asset_b, seed)
      .expect("cross-Gamma inputs are unsupported; use try_cross_gamma_with_seed")
  }

  /// Fallible deterministic cross-Gamma using common random numbers.
  pub fn try_cross_gamma_with_seed(
    &self,
    payoff: &MtPayoff<T>,
    asset_a: usize,
    asset_b: usize,
    seed: u64,
  ) -> anyhow::Result<T> {
    let dimension = self.params.n_assets();
    if asset_a >= dimension {
      anyhow::bail!("asset_a={asset_a} is out of bounds for n_assets={dimension}");
    }
    if asset_b >= dimension {
      anyhow::bail!("asset_b={asset_b} is out of bounds for n_assets={dimension}");
    }
    let spot = self.params.assets[asset_b].s0;
    let bump = spot * T::from_f64_fast(0.01);
    if bump <= T::zero() {
      anyhow::bail!("asset_b spot is too small for a representable central-difference bump");
    }
    let mut up = self.params.clone();
    up.assets[asset_b].s0 += bump;
    let mut dn = self.params.clone();
    dn.assets[asset_b].s0 -= bump;

    let d_up =
      MtGreeks::try_new(up, self.h, self.n_paths)?.try_delta_with_seed(payoff, asset_a, seed)?;
    let d_dn =
      MtGreeks::try_new(dn, self.h, self.n_paths)?.try_delta_with_seed(payoff, asset_a, seed)?;
    Ok((d_up - d_dn) / (bump + bump))
  }

  /// Backward-compatible alias retained for the integration tests.
  pub fn cross_gamma_fd(&self, payoff: &MtPayoff<T>, asset_a: usize, asset_b: usize) -> T {
    self.cross_gamma(payoff, asset_a, asset_b)
  }

  /// Vega `dPrice/dSigma` via finite difference on the initial volatility.
  ///
  /// Uses a fixed-seed common-random-numbers bump internally to reduce Monte
  /// Carlo variance. This is a numerical utility, not an M-T weight estimator.
  pub fn vega(&self, payoff: &MtPayoff<T>, asset: usize) -> T {
    self.vega_with_seed(payoff, asset, 0x9E37_79B9_7F4A_7C15_u64 ^ asset as u64)
  }

  /// Deterministic Vega estimator using common random numbers for the
  /// up/down finite-difference bump.
  pub fn vega_with_seed(&self, payoff: &MtPayoff<T>, asset: usize, seed: u64) -> T {
    assert!(asset < self.params.n_assets(), "asset index out of bounds");
    let v0 = self.params.assets[asset].v0;
    assert!(v0 >= T::zero(), "initial variance must be non-negative");
    let sigma = <T as num_traits::Float>::sqrt(v0);
    let bump = sigma.max(T::from_f64_fast(0.01)) * T::from_f64_fast(0.01);
    let sigma_up = sigma + bump;
    let sigma_dn = (sigma - bump).max(T::zero());
    let mut up = self.params.clone();
    up.assets[asset].v0 = sigma_up * sigma_up;
    let mut dn = self.params.clone();
    dn.assets[asset].v0 = sigma_dn * sigma_dn;

    let p_up = MtGreeks::new(up, self.h, self.n_paths).price_with_seed(payoff, seed);
    let p_dn = MtGreeks::new(dn, self.h, self.n_paths).price_with_seed(payoff, seed);
    (p_up - p_dn) / (sigma_up - sigma_dn)
  }

  /// Initial-variance sensitivity `dPrice/dv0`.
  pub fn vega_v0(&self, payoff: &MtPayoff<T>, asset: usize) -> T {
    self.vega_v0_with_seed(payoff, asset, 0x517C_C1B7_2722_0A95_u64 ^ asset as u64)
  }

  /// Deterministic initial-variance sensitivity using common random numbers.
  pub fn vega_v0_with_seed(&self, payoff: &MtPayoff<T>, asset: usize, seed: u64) -> T {
    assert!(asset < self.params.n_assets(), "asset index out of bounds");
    let v0 = self.params.assets[asset].v0;
    assert!(v0 >= T::zero(), "initial variance must be non-negative");
    let bump = v0.max(T::from_f64_fast(1e-4)) * T::from_f64_fast(0.01);
    let v_up = v0 + bump;
    let v_dn = (v0 - bump).max(T::zero());
    let mut up = self.params.clone();
    up.assets[asset].v0 = v_up;
    let mut dn = self.params.clone();
    dn.assets[asset].v0 = v_dn;

    let p_up = MtGreeks::new(up, self.h, self.n_paths).price_with_seed(payoff, seed);
    let p_dn = MtGreeks::new(dn, self.h, self.n_paths).price_with_seed(payoff, seed);
    (p_up - p_dn) / (v_up - v_dn)
  }
}
