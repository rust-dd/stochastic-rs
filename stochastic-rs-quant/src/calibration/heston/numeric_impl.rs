use nalgebra::DMatrix;
use nalgebra::DVector;
use stochastic_rs_stats::heston_mle::nmle_heston;
use stochastic_rs_stats::heston_mle::nmle_heston_with_delta;
use stochastic_rs_stats::heston_mle::pmle_heston;
use stochastic_rs_stats::heston_mle::pmle_heston_with_delta;
use stochastic_rs_stats::heston_nml_cekf::nmle_cekf_heston;

use super::calibrator::HestonCalibrator;
use super::params::HestonJacobianMethod;
use super::params::HestonMleSeedMethod;
use super::params::HestonParams;
use super::transform::canonicalize;
use super::transform::from_optimizer_coordinates;
use super::transform::to_optimizer_coordinates;
use crate::OptionType;
use crate::pricing::heston::HestonPricer;
use crate::traits::PricerExt;

impl HestonCalibrator {
  pub(super) fn fallback_params() -> HestonParams {
    HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.5,
      rho: -0.5,
    }
    .projected()
  }

  pub(super) fn inferred_mle_delta(&self, n_obs: usize) -> f64 {
    if let Some(dt) = self.mle_delta
      && dt.is_finite()
      && dt > 0.0
    {
      return dt;
    }
    1.0 / n_obs.saturating_sub(1).max(1) as f64
  }

  /// Build an initial parameter guess from historical series via NMLE/PMLE/NMLE-CEKF.
  ///
  /// Source:
  /// - Wang et al. (2018)
  ///   https://doi.org/10.1007/s11432-017-9215-8
  ///   http://scis.scichina.com/en/2018/042202.pdf
  pub(super) fn infer_initial_guess_from_series(&self) -> Option<HestonParams> {
    match self.mle_seed_method {
      HestonMleSeedMethod::Nmle => {
        let (s, v, r) = (self.mle_s.clone()?, self.mle_v.clone()?, self.mle_r?);
        if s.len() < 2 || v.len() < 2 || s.len() != v.len() {
          return None;
        }
        let p: HestonParams = if self.mle_delta.is_some() {
          let delta = self.inferred_mle_delta(s.len());
          nmle_heston_with_delta(s.view(), v.view(), r, delta).into()
        } else {
          nmle_heston(s.view(), v.view(), r).into()
        };
        Some(p.projected())
      }
      HestonMleSeedMethod::Pmle => {
        let (s, v, r) = (self.mle_s.clone()?, self.mle_v.clone()?, self.mle_r?);
        if s.len() < 2 || v.len() < 2 || s.len() != v.len() {
          return None;
        }
        let p: HestonParams = if self.mle_delta.is_some() {
          let delta = self.inferred_mle_delta(s.len());
          pmle_heston_with_delta(s.view(), v.view(), r, delta).into()
        } else {
          pmle_heston(s.view(), v.view(), r).into()
        };
        Some(p.projected())
      }
      HestonMleSeedMethod::NmleCekf => {
        let (s, r) = (self.mle_s.clone()?, self.mle_r?);
        if s.len() < 2 {
          return None;
        }
        let mut cfg = self.nmle_cekf_config.clone().unwrap_or_default();
        cfg.r = r;
        cfg.delta = self.inferred_mle_delta(s.len());
        if let Some(v_ts) = self.mle_v.as_ref()
          && !v_ts.is_empty()
          && v_ts[0].is_finite()
          && v_ts[0] > 0.0
        {
          cfg.initial_v0 = v_ts[0];
        }
        let out = nmle_cekf_heston(s.view(), cfg);
        let p: HestonParams = out.params.into();
        Some(p.projected())
      }
    }
  }

  pub(super) fn ensure_initial_guess(&mut self) {
    if self.params.is_none() {
      self.params = Some(
        self
          .infer_initial_guess_from_series()
          .unwrap_or_else(Self::fallback_params),
      );
    }
  }

  pub(super) fn effective_params(&self) -> HestonParams {
    let projected = if let Some(p) = &self.params {
      p.clone().projected()
    } else {
      self
        .infer_initial_guess_from_series()
        .unwrap_or_else(Self::fallback_params)
    };
    canonicalize(&projected)
  }

  pub(super) fn compute_model_prices_for_numeric(&self, params: &HestonParams) -> DVector<f64> {
    let mut c_model = DVector::zeros(self.c_market.len());

    for (idx, _) in self.c_market.iter().enumerate() {
      let pricer = HestonPricer::new(
        self.s[idx],
        params.v0,
        self.k[idx],
        self.r,
        self.q,
        params.rho,
        params.kappa,
        params.theta,
        params.sigma,
        Some(0.0),
        Some(self.flat_t[idx]),
        None,
        None,
      );
      let (call, put) = pricer.calculate_call_put();

      match self.option_type {
        OptionType::Call => c_model[idx] = call.max(0.0),
        OptionType::Put => c_model[idx] = put.max(0.0),
      }
    }

    c_model
  }

  pub(super) fn compute_model_prices_for(&self, params: &HestonParams) -> DVector<f64> {
    match self.jacobian_method {
      HestonJacobianMethod::NumericFiniteDiff => self.compute_model_prices_for_numeric(params),
      HestonJacobianMethod::CuiAnalytic => self
        .compute_model_prices_for_cui(params)
        .unwrap_or_else(|| self.compute_model_prices_for_numeric(params)),
    }
  }

  pub(super) fn residuals_for(&self, params: &HestonParams) -> DVector<f64> {
    (self.c_market.clone() - self.compute_model_prices_for(params))
      .component_mul(&self.residual_weights)
  }

  /// Numerically approximate the residual Jacobian with respect to physical parameters.
  #[cfg(test)]
  #[allow(non_snake_case)]
  pub(super) fn numeric_jacobian(&self, params: &HestonParams) -> DMatrix<f64> {
    self.finite_difference_jacobian(params, false)
  }

  pub(super) fn numeric_optimizer_jacobian(&self, params: &HestonParams) -> DMatrix<f64> {
    self.finite_difference_jacobian(params, true)
  }

  #[allow(non_snake_case)]
  fn finite_difference_jacobian(
    &self,
    params: &HestonParams,
    optimizer_denominator: bool,
  ) -> DMatrix<f64> {
    let n = self.c_market.len();
    let p = 5usize;
    let coordinates = to_optimizer_coordinates(params);
    let mut J = DMatrix::zeros(n, p);

    for col in 0..p {
      let h = 1e-5 * (1.0 + coordinates[col].abs());
      let mut plus = coordinates.clone();
      let mut minus = coordinates.clone();
      plus[col] += h;
      minus[col] -= h;
      let params_plus = from_optimizer_coordinates(&plus);
      let params_minus = from_optimizer_coordinates(&minus);

      let r_plus = self.residuals_for(&params_plus);
      let r_minus = self.residuals_for(&params_minus);
      let denominator = if optimizer_denominator {
        2.0 * h
      } else {
        let plus_physical = DVector::from(params_plus);
        let minus_physical = DVector::from(params_minus);
        plus_physical[col] - minus_physical[col]
      };
      let diff = (r_plus - r_minus) / denominator;
      for row in 0..n {
        J[(row, col)] = diff[row];
      }
    }

    J
  }
}
