use stochastic_rs_core::simd_rng::SeedExt;

use super::greeks::MtGreeks;
use super::payoff::MtPayoff;
use crate::QuantError;
use crate::traits::FloatExt;

impl<T: FloatExt, S: SeedExt> MtGreeks<T, S> {
  /// Cross-gamma by a common-random-numbers finite difference on Delta, not a higher-order
  /// Malliavin–Thalmaier weight.
  pub fn cross_gamma(&self, payoff: &MtPayoff<T>, asset_a: usize, asset_b: usize) -> T {
    self
      .try_cross_gamma(payoff, asset_a, asset_b)
      .unwrap_or_else(|e| panic!("{e}"))
  }

  /// Fallible common-random-numbers cross-Gamma.
  pub fn try_cross_gamma(
    &self,
    payoff: &MtPayoff<T>,
    asset_a: usize,
    asset_b: usize,
  ) -> Result<T, QuantError> {
    let dimension = self.params.n_assets();
    for (what, asset) in [("asset_a", asset_a), ("asset_b", asset_b)] {
      if asset >= dimension {
        return Err(QuantError::invalid(
          what,
          format!("must be below n_assets = {dimension}, got {asset}"),
        ));
      }
    }
    let spot = self.params.assets[asset_b].s;
    let bump = spot * T::from_f64_fast(0.01);
    if bump <= T::zero() {
      return Err(QuantError::Numerical {
        what: "the cross-Gamma spot bump",
        reason: format!("1% of asset_b's spot {spot:?} is not a representable positive bump"),
      });
    }
    let mut up = self.params.clone();
    up.assets[asset_b].s += bump;
    let mut dn = self.params.clone();
    dn.assets[asset_b].s -= bump;

    let seeds = self.path_seeds();
    let d_up =
      MtGreeks::try_new(up, self.h, self.n_paths)?.try_delta_at(payoff, asset_a, &seeds)?;
    let d_dn =
      MtGreeks::try_new(dn, self.h, self.n_paths)?.try_delta_at(payoff, asset_a, &seeds)?;
    Ok((d_up - d_dn) / (bump + bump))
  }

  /// `dPrice/dSigma` by a central bump of the initial volatility on common random numbers, a
  /// numerical utility rather than an M-T weight estimator.
  pub fn vega(&self, payoff: &MtPayoff<T>, asset: usize) -> T {
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

    let seeds = self.path_seeds();
    let p_up = MtGreeks::new(up, self.h, self.n_paths).price_at(payoff, &seeds);
    let p_dn = MtGreeks::new(dn, self.h, self.n_paths).price_at(payoff, &seeds);
    (p_up - p_dn) / (sigma_up - sigma_dn)
  }

  /// `dPrice/dv0` by a central bump of `v0` on common random numbers.
  pub fn vega_v0(&self, payoff: &MtPayoff<T>, asset: usize) -> T {
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

    let seeds = self.path_seeds();
    let p_up = MtGreeks::new(up, self.h, self.n_paths).price_at(payoff, &seeds);
    let p_dn = MtGreeks::new(dn, self.h, self.n_paths).price_at(payoff, &seeds);
    (p_up - p_dn) / (v_up - v_dn)
  }
}
