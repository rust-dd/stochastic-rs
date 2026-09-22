use anyhow::Result;
use anyhow::bail;
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_distributions::traits::Grid2D;

use super::result::HestonSlvCalibrationResult;
use super::result::HestonSlvFit;
use crate::CalibrationLossScore;
use crate::OptionType;
use crate::calibration::heston::HestonCalibrationResult;
use crate::calibration::heston::HestonCalibrator;
use crate::calibration::heston::HestonParams;
use crate::calibration::levy::MarketSlice;
use crate::pricing::dupire::Dupire;
use crate::pricing::slv::FokkerPlanckMethod;
use crate::pricing::slv::HestonSlvParams;
use crate::pricing::slv::LeverageSurface;
use crate::pricing::slv::ParticleMethod;
use crate::pricing::slv::calibrate_leverage;
use crate::pricing::slv::calibrate_leverage_fokker_planck;
use crate::traits::Calibrator;

/// Largest deviation between a requested rate and the calibration rate that
/// still counts as the same rate — the pricer's own match tolerance.
pub(super) const RATE_MATCH_TOL: f64 = 1e-12;

/// How the leverage surface is read off the local volatility: the
/// Guyon–Henry-Labordère particle cloud, or the finite-volume solution of
/// the forward Kolmogorov equation after Wyns & Du Toit. Both reproduce the
/// vanilla surface; the cloud is cheap and noisy, the PDE deterministic and
/// smooth, and the fit each reports is read off its own density.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum LeverageMethod {
  Particle(ParticleMethod),
  FokkerPlanck(FokkerPlanckMethod),
}

impl Default for LeverageMethod {
  fn default() -> Self {
    Self::Particle(ParticleMethod::default())
  }
}

/// Heston SLV calibrator: a vanilla call surface in, a leverage-calibrated
/// model out — the module doc has the three steps.
///
/// The input is a call-price grid, `calls[[j, i]] = C(K_i, T_j)`, the shape
/// [`Dupire`] and the vol-surface pipeline take. Everything else is a
/// builder knob with a default: the mixing fraction (`1`), the Heston
/// parameters (fitted unless pinned), the local volatility (Dupire on the
/// input calls unless supplied), the Dupire denominator floor, and the
/// [`LeverageMethod`] (the particle cloud unless
/// [`with_fokker_planck`](Self::with_fokker_planck) picks the PDE).
///
/// Source:
/// - Guyon & Henry-Labordère (2012), *Being particular about calibration*,
///   Risk 25(1)
/// - Cozma, Mariapragassam & Reisinger (2017), arXiv:1701.06001, §3.4
/// - Wyns & Du Toit (2016), arXiv:1611.02961, §4
#[derive(Clone, Debug)]
pub struct HestonSlvCalibrator {
  /// Spot.
  pub s: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Strikes, ascending.
  pub strikes: Vec<f64>,
  /// Maturities in years, ascending.
  pub maturities: Vec<f64>,
  /// Present call prices, shape `(maturities.len(), strikes.len())`.
  pub calls: Array2<f64>,
  /// Mixing fraction $\eta \in [0, 1]$.
  pub eta: f64,
  /// Pinned Heston parameters; `None` fits them to the calls.
  pub heston_params: Option<HestonParams>,
  /// Starting point of the Heston fit; `None` lets
  /// [`HestonCalibrator`] pick its own.
  pub heston_initial_guess: Option<HestonParams>,
  /// A local-volatility surface on the input grid; `None` reads it off the
  /// calls with [`Dupire`].
  pub local_vol: Option<Array2<f64>>,
  /// [`Dupire::eps`], the floor of the Dupire denominator.
  pub dupire_eps: f64,
  /// How the leverage is calibrated and tuned.
  pub method: LeverageMethod,
}

impl HestonSlvCalibrator {
  pub fn new(
    s: f64,
    r: f64,
    q: f64,
    strikes: Vec<f64>,
    maturities: Vec<f64>,
    calls: Array2<f64>,
  ) -> Self {
    Self {
      s,
      r,
      q,
      strikes,
      maturities,
      calls,
      eta: 1.0,
      heston_params: None,
      heston_initial_guess: None,
      local_vol: None,
      dupire_eps: 1e-6,
      method: LeverageMethod::default(),
    }
  }

  /// The mixing fraction $\eta$: `0` is the pure local-volatility model, `1`
  /// the Heston dynamics under a leverage correction.
  pub fn with_mixing(mut self, eta: f64) -> Self {
    self.eta = eta;
    self
  }

  /// Pin the Heston parameters instead of fitting them.
  pub fn with_heston_params(mut self, params: HestonParams) -> Self {
    self.heston_params = Some(params);
    self
  }

  /// The starting point of the Heston fit, when the parameters are not
  /// pinned.
  pub fn with_heston_initial_guess(mut self, guess: HestonParams) -> Self {
    self.heston_initial_guess = Some(guess);
    self
  }

  /// Supply the local volatility $\sigma_{\text{LV}}(T_j, K_i)$ on the input
  /// grid instead of reading it off the calls with [`Dupire`].
  pub fn with_local_vol(mut self, local_vol: Array2<f64>) -> Self {
    self.local_vol = Some(local_vol);
    self
  }

  /// The Dupire denominator floor, [`Dupire::eps`].
  pub fn with_dupire_eps(mut self, eps: f64) -> Self {
    self.dupire_eps = eps;
    self
  }

  /// Calibrate the leverage by the particle method with this tuning.
  pub fn with_particle_method(mut self, method: ParticleMethod) -> Self {
    self.method = LeverageMethod::Particle(method);
    self
  }

  /// Calibrate the leverage by the forward Kolmogorov equation with this
  /// tuning.
  pub fn with_fokker_planck(mut self, method: FokkerPlanckMethod) -> Self {
    self.method = LeverageMethod::FokkerPlanck(method);
    self
  }

  /// The input grid as one [`MarketSlice`] per maturity, all calls — what
  /// the Heston fit consumes.
  pub fn market_slices(&self) -> Vec<MarketSlice> {
    self
      .maturities
      .iter()
      .enumerate()
      .map(|(j, &tau)| MarketSlice {
        strikes: self.strikes.clone(),
        prices: self.calls.row(j).to_vec(),
        is_call: vec![true; self.strikes.len()],
        tau,
      })
      .collect()
  }

  fn validate(&self) -> Result<()> {
    let (nt, nk) = (self.maturities.len(), self.strikes.len());
    if nk < 3 {
      bail!("at least three strikes are needed for the Dupire stencil, got {nk}");
    }
    if nt == 0 {
      bail!("at least one maturity is needed");
    }
    if self.local_vol.is_none() && nt < 2 {
      bail!(
        "the Dupire time derivative needs at least two maturities; add one or supply the local \
         volatility with `with_local_vol`"
      );
    }
    if !(self.s.is_finite() && self.s > 0.0) {
      bail!("the spot must be finite and positive, got {}", self.s);
    }
    if !(self.r.is_finite() && self.q.is_finite()) {
      bail!(
        "the rates must be finite, got r = {}, q = {}",
        self.r,
        self.q
      );
    }
    if !(self.eta.is_finite() && (0.0..=1.0).contains(&self.eta)) {
      bail!(
        "the mixing fraction eta must lie in [0, 1], got {}",
        self.eta
      );
    }
    if self.strikes.iter().any(|k| !(k.is_finite() && *k > 0.0)) {
      bail!("strikes must be finite and positive");
    }
    if self.strikes.windows(2).any(|w| w[0] >= w[1]) {
      bail!("strikes must be strictly ascending");
    }
    if self.maturities.iter().any(|t| !(t.is_finite() && *t > 0.0)) {
      bail!("maturities must be finite and positive");
    }
    if self.maturities.windows(2).any(|w| w[0] >= w[1]) {
      bail!("maturities must be strictly ascending");
    }
    if self.calls.dim() != (nt, nk) {
      bail!(
        "calls must have shape (maturities, strikes) = ({nt}, {nk}), got {:?}",
        self.calls.dim()
      );
    }
    if self.calls.iter().any(|c| !(c.is_finite() && *c >= 0.0)) {
      bail!("call prices must be finite and non-negative");
    }
    if let Some(lv) = &self.local_vol
      && lv.dim() != (nt, nk)
    {
      bail!(
        "the local volatility must have shape (maturities, strikes) = ({nt}, {nk}), got {:?}",
        lv.dim()
      );
    }
    if !(self.dupire_eps.is_finite() && self.dupire_eps > 0.0) {
      bail!(
        "dupire_eps must be finite and positive, got {}",
        self.dupire_eps
      );
    }
    Ok(())
  }

  /// The Heston parameters and, when they were fitted, the fit.
  fn heston(
    &self,
    initial: Option<HestonParams>,
  ) -> Result<(HestonParams, Option<HestonCalibrationResult>)> {
    if let Some(pinned) = &self.heston_params {
      return Ok((pinned.clone(), None));
    }
    let slices = self.market_slices();
    let calibrator = HestonCalibrator::from_slices(
      initial.or_else(|| self.heston_initial_guess.clone()),
      &slices,
      self.s,
      self.r,
      Some(self.q),
      OptionType::Call,
      false,
    );
    let fit = calibrator.calibrate(None)?;
    Ok((fit.params.clone(), Some(fit)))
  }

  /// The local volatility as the `(t, K)` grid the particle method reads.
  fn local_vol_grid(&self) -> Result<Grid2D<f64>> {
    let raw = match &self.local_vol {
      Some(lv) => lv.clone(),
      None => Dupire::builder(
        self.strikes.clone(),
        self.maturities.clone(),
        self.calls.clone(),
      )
      .r(self.r)
      .q(self.q)
      .eps(self.dupire_eps)
      .build()
      .local_vol_surface(),
    };
    let cleaned = clean_local_vol(raw, &self.maturities)?;
    Ok(Grid2D::new(
      Array1::from_vec(self.maturities.clone()),
      Array1::from_vec(self.strikes.clone()),
      cleaned,
    ))
  }
}

impl HestonSlvCalibrator {
  /// The leverage by the chosen method, and the input grid repriced from
  /// that method's own density — the particle cloud's payoff averages, or
  /// the trapezoid rule over the finite-volume marginal — row-major in the
  /// maturities like `calls`.
  fn leverage_and_model_prices(
    &self,
    params: &HestonSlvParams,
    local_vol: &Grid2D<f64>,
  ) -> Result<(LeverageSurface, Vec<f64>)> {
    let mut model = Vec::with_capacity(self.maturities.len() * self.strikes.len());
    match &self.method {
      LeverageMethod::Particle(method) => {
        let run = calibrate_leverage(
          params,
          self.s,
          self.r,
          self.q,
          local_vol,
          &self.maturities,
          method,
        )?;
        for (j, cloud) in run.snapshots.iter().enumerate() {
          let discount = (-self.r * self.maturities[j]).exp();
          let paths = cloud.len() as f64;
          for &k in &self.strikes {
            model.push(discount * cloud.iter().map(|s| (s - k).max(0.0)).sum::<f64>() / paths);
          }
        }
        Ok((run.leverage, model))
      }
      LeverageMethod::FokkerPlanck(method) => {
        let run = calibrate_leverage_fokker_planck(
          params,
          self.s,
          self.r,
          self.q,
          local_vol,
          &self.maturities,
          method,
        )?;
        for (j, &tau) in self.maturities.iter().enumerate() {
          for &k in &self.strikes {
            model.push(run.density.call_price(j, k, self.r, tau));
          }
        }
        Ok((run.leverage, model))
      }
    }
  }
}

/// The Dupire surface with its `NaN` cells filled: the stencil-boundary
/// columns and any leading or trailing run take the nearest admissible cell
/// of their row, an interior hole is interpolated linearly in strike between
/// the admissible cells around it. A row with no admissible cell at all is
/// an error — there is no local volatility to calibrate to at that maturity.
pub(super) fn clean_local_vol(mut surface: Array2<f64>, maturities: &[f64]) -> Result<Array2<f64>> {
  let (nt, nk) = surface.dim();
  for j in 0..nt {
    let finite = (0..nk)
      .filter(|&i| surface[[j, i]].is_finite())
      .collect::<Vec<_>>();
    let (Some(&first), Some(&last)) = (finite.first(), finite.last()) else {
      bail!(
        "the local volatility has no admissible cell at maturity {}: the call slice is not \
         locally arbitrage-free",
        maturities[j]
      );
    };
    for i in 0..first {
      surface[[j, i]] = surface[[j, first]];
    }
    for i in last + 1..nk {
      surface[[j, i]] = surface[[j, last]];
    }
    for pair in finite.windows(2) {
      let (a, b) = (pair[0], pair[1]);
      for i in a + 1..b {
        let w = (i - a) as f64 / (b - a) as f64;
        surface[[j, i]] = surface[[j, a]] + w * (surface[[j, b]] - surface[[j, a]]);
      }
    }
  }
  Ok(surface)
}

impl Calibrator for HestonSlvCalibrator {
  /// The starting point of the Heston fit; ignored when the parameters are
  /// pinned with [`with_heston_params`](HestonSlvCalibrator::with_heston_params).
  type InitialGuess = HestonParams;
  type Params = HestonSlvFit;
  type Output = HestonSlvCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    self.validate()?;
    let (heston, heston_fit) = self.heston(initial)?;
    let params = HestonSlvParams {
      kappa: heston.kappa,
      theta: heston.theta,
      sigma: heston.sigma,
      rho: heston.rho,
      v0: heston.v0,
      eta: self.eta,
    };
    let local_vol = self.local_vol_grid()?;
    let (leverage, model) = self.leverage_and_model_prices(&params, &local_vol)?;
    let market = self.calls.iter().copied().collect::<Vec<_>>();
    let loss = CalibrationLossScore::compute(&market, &model);
    let max_error = market
      .iter()
      .zip(model.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0, f64::max);
    let leverage_finite = leverage.values().iter().all(|l| l.is_finite());
    let converged =
      heston_fit.as_ref().is_none_or(|f| f.converged) && leverage_finite && max_error.is_finite();

    Ok(HestonSlvCalibrationResult {
      fit: HestonSlvFit { params, leverage },
      heston: heston_fit,
      loss,
      max_error,
      converged,
      s: self.s,
      r: self.r,
      q: self.q,
    })
  }
}
