use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_distributions::traits::Grid2D;

use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::Knob;
use crate::calibration::heston::HestonCalibrator;
use crate::calibration::quote::assert_spot_and_rates;
use crate::error::check_grid_shape;
use crate::pricing::dupire::Dupire;
use crate::pricing::heston::HestonParams;
use crate::pricing::slv::FokkerPlanckMethod;
use crate::pricing::slv::HestonSlvParams;
use crate::pricing::slv::HestonSlvPricer;
use crate::pricing::slv::LeverageSurface;
use crate::pricing::slv::ParticleMethod;
use crate::pricing::slv::calibrate_leverage;
use crate::pricing::slv::calibrate_leverage_fokker_planck;
use crate::traits::Calibrator;
use crate::traits::ToModel;

/// The calibrated model: the parameters and the leverage that makes them reprice the surface.
#[derive(Clone, Debug)]
pub struct HestonSlvFit {
  pub params: HestonSlvParams,
  pub leverage: LeverageSurface,
  /// The rates the leverage is anchored to.
  pub r: f64,
  pub q: f64,
}

impl ToModel for Calibrated<HestonSlvFit, Option<Calibrated<HestonParams>>> {
  type Model = HestonSlvPricer;

  fn to_model(&self) -> HestonSlvPricer {
    let fit = &self.params;
    HestonSlvPricer::new(fit.params, fit.leverage.clone(), fit.r, fit.q)
  }
}

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
  /// A local-volatility surface on the input grid; `None` reads it off the
  /// calls with [`Dupire`].
  pub local_vol: Option<Array2<f64>>,
  /// [`Dupire::eps`], the floor of the Dupire denominator.
  pub dupire_eps: f64,
  /// How the leverage is calibrated and tuned.
  pub method: LeverageMethod,
  /// The Heston fit's knobs, its trail in `diagnostics`; the grid reads `loss_metrics`, a pinned
  /// fit refuses the rest, and `regularization` is `Unsupported`.
  pub options: CalibrationOptions,
}

impl HestonSlvCalibrator {
  /// Panics unless `s` is finite and positive and `r`, `q` finite; `calibrate` checks the grid.
  pub fn new(
    s: f64,
    r: f64,
    q: f64,
    strikes: Vec<f64>,
    maturities: Vec<f64>,
    calls: Array2<f64>,
  ) -> Self {
    assert_spot_and_rates(s, r, q);
    Self {
      s,
      r,
      q,
      strikes,
      maturities,
      calls,
      eta: 1.0,
      heston_params: None,
      local_vol: None,
      dupire_eps: 1e-6,
      method: LeverageMethod::default(),
      options: CalibrationOptions::default(),
    }
  }

  /// The mixing fraction $\eta$: `0` is pure local volatility, `1` Heston under a leverage
  /// correction; panics unless `eta` lies in `[0, 1]`.
  pub fn with_mixing(mut self, eta: f64) -> Self {
    assert!(
      (0.0..=1.0).contains(&eta),
      "eta must satisfy `(0.0..=1.0).contains(&eta)`, got eta = {eta:?}"
    );
    self.eta = eta;
    self
  }

  /// Pin the Heston parameters instead of fitting them.
  pub fn with_heston_params(mut self, params: HestonParams) -> Self {
    self.heston_params = Some(params);
    self
  }

  /// Supply the local volatility $\sigma_{\text{LV}}(T_j, K_i)$ on the input
  /// grid instead of reading it off the calls with [`Dupire`].
  ///
  /// Prefer this whenever a smooth surface exists — an SSVI fit, an
  /// analytic Black surface: [`Dupire`]'s read of a call grid is a
  /// finite-difference estimate, and against QuantLib's local volatility of
  /// the same Heston market it carries a 1–3 % error that dominates the
  /// leverage's (the calibration's in-sample RMSE is 0.036 on a spot of 100
  /// from the finite-difference read and 0.004 from the smooth surface, with
  /// the leverage then within 0.01 of QuantLib's own SLV model).
  pub fn with_local_vol(mut self, local_vol: Array2<f64>) -> Self {
    self.local_vol = Some(local_vol);
    self
  }

  /// The Dupire denominator floor, [`Dupire::eps`]; panics unless `eps` is finite and positive.
  pub fn with_dupire_eps(mut self, eps: f64) -> Self {
    assert!(
      eps.is_finite() && eps > 0.0,
      "eps must satisfy `eps.is_finite() && eps > 0.0`, got eps = {eps:?}"
    );
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

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }

  /// The call grid as quotes, maturity by maturity: the Heston fit's input.
  fn quotes(&self) -> Vec<OptionQuote> {
    self
      .calls
      .indexed_iter()
      .map(|((j, i), &call)| {
        OptionQuote::new(self.strikes[i], self.maturities[j], call, OptionType::Call)
      })
      .collect()
  }

  fn validate(&self) -> Result<(), QuantError> {
    let (nt, nk) = (self.maturities.len(), self.strikes.len());
    if nk < 3 {
      return Err(QuantError::InsufficientData {
        what: "strikes in the Dupire stencil",
        needed: 3,
        got: nk,
      });
    }
    if nt == 0 {
      return Err(QuantError::InsufficientData {
        what: "maturities in the grid",
        needed: 1,
        got: 0,
      });
    }
    if self.local_vol.is_none() && nt < 2 {
      return Err(QuantError::InsufficientData {
        what: "maturities for the Dupire time derivative without `with_local_vol`",
        needed: 2,
        got: nt,
      });
    }
    if !(self.s.is_finite() && self.s > 0.0) {
      return Err(QuantError::invalid(
        "spot",
        format!("must be finite and positive, got {}", self.s),
      ));
    }
    if !(self.r.is_finite() && self.q.is_finite()) {
      return Err(QuantError::invalid(
        "rates",
        format!("must be finite, got r = {}, q = {}", self.r, self.q),
      ));
    }
    if !(self.eta.is_finite() && (0.0..=1.0).contains(&self.eta)) {
      return Err(QuantError::invalid(
        "mixing fraction eta",
        format!("must lie in [0, 1], got {}", self.eta),
      ));
    }
    if let Some((i, k)) = first_where(&self.strikes, |k| !(k.is_finite() && k > 0.0)) {
      return Err(QuantError::invalid(
        "strikes",
        format!("strike {i} must be finite and positive, got {k}"),
      ));
    }
    if let Some(i) = self.strikes.windows(2).position(|w| w[0] >= w[1]) {
      return Err(QuantError::invalid(
        "strikes",
        format!(
          "must be strictly ascending, got strike {i} = {} before {}",
          self.strikes[i],
          self.strikes[i + 1]
        ),
      ));
    }
    if let Some((j, t)) = first_where(&self.maturities, |t| !(t.is_finite() && t > 0.0)) {
      return Err(QuantError::invalid(
        "maturities",
        format!("maturity {j} must be finite and positive, got {t}"),
      ));
    }
    if let Some(j) = self.maturities.windows(2).position(|w| w[0] >= w[1]) {
      return Err(QuantError::invalid(
        "maturities",
        format!(
          "must be strictly ascending, got maturity {j} = {} before {}",
          self.maturities[j],
          self.maturities[j + 1]
        ),
      ));
    }
    check_grid_shape(
      [
        "call rows, one per maturity",
        "call columns, one per strike",
      ],
      (nt, nk),
      self.calls.dim(),
    )?;
    if let Some(((j, i), c)) = self
      .calls
      .indexed_iter()
      .find(|(_, c)| !(c.is_finite() && **c >= 0.0))
    {
      return Err(QuantError::invalid(
        "call prices",
        format!("calls[[{j}, {i}]] must be finite and non-negative, got {c}"),
      ));
    }
    if let Some(lv) = &self.local_vol {
      check_grid_shape(
        [
          "local-volatility rows, one per maturity",
          "local-volatility columns, one per strike",
        ],
        (nt, nk),
        lv.dim(),
      )?;
    }
    if !(self.dupire_eps.is_finite() && self.dupire_eps > 0.0) {
      return Err(QuantError::invalid(
        "dupire_eps",
        format!("must be finite and positive, got {}", self.dupire_eps),
      ));
    }
    Ok(())
  }

  /// The Heston parameters and, when they were fitted, the fit.
  fn heston(
    &self,
    initial: Option<HestonParams>,
  ) -> Result<(HestonParams, Option<Calibrated<HestonParams>>), QuantError> {
    if let Some(pinned) = self.heston_params {
      return Ok((pinned, None));
    }
    let fit = HestonCalibrator::new(self.s, self.r, self.q, self.quotes())
      .with_options(self.options.clone())
      .calibrate(initial)?;
    Ok((fit.params, Some(fit)))
  }

  /// The local volatility as the `(t, K)` grid the particle method reads.
  fn local_vol_grid(&self) -> Result<Grid2D<f64>, QuantError> {
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
  ) -> Result<(LeverageSurface, Vec<f64>), QuantError> {
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
pub(super) fn clean_local_vol(
  mut surface: Array2<f64>,
  maturities: &[f64],
) -> Result<Array2<f64>, QuantError> {
  let (nt, nk) = surface.dim();
  for j in 0..nt {
    let finite = (0..nk)
      .filter(|&i| surface[[j, i]].is_finite())
      .collect::<Vec<_>>();
    let (Some(&first), Some(&last)) = (finite.first(), finite.last()) else {
      return Err(QuantError::invalid(
        "call slice",
        format!(
          "the local volatility has no admissible cell at maturity {} (the slice is not locally \
           arbitrage-free)",
          maturities[j]
        ),
      ));
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

fn first_where(values: &[f64], bad: impl Fn(f64) -> bool) -> Option<(usize, f64)> {
  values.iter().copied().enumerate().find(|&(_, v)| bad(v))
}

impl Calibrator for HestonSlvCalibrator {
  type Params = HestonSlvFit;
  type Output = Calibrated<HestonSlvFit, Option<Calibrated<HestonParams>>>;
  type Error = QuantError;

  /// The Heston fit, `diagnostics` (`None` when pinned), starts from `initial.params.heston`; the
  /// errors are the grid's, the Heston fit's, the leverage's and the `CalibrationOptions` ones.
  fn calibrate(&self, initial: Option<HestonSlvFit>) -> Result<Self::Output, QuantError> {
    self.validate()?;
    let unread: &[Knob] = match self.heston_params {
      Some(_) => &[
        Knob::MaxIterations,
        Knob::Tolerance,
        Knob::Patience,
        Knob::RecordHistory,
        Knob::Regularization,
      ],
      None => &[Knob::Regularization],
    };
    self.options.check(unread)?;
    let start = initial.map(|fit| fit.params.heston);
    let (heston, heston_fit) = self.heston(start)?;
    let params = HestonSlvParams {
      heston,
      eta: self.eta,
    };
    let local_vol = self.local_vol_grid()?;
    let (leverage, model) = self.leverage_and_model_prices(&params, &local_vol)?;
    let market = self.calls.iter().copied().collect::<Vec<_>>();
    let leverage_finite = leverage.values().iter().all(|l| l.is_finite());
    let fit = HestonSlvFit {
      params,
      leverage,
      r: self.r,
      q: self.q,
    };
    let mut fit = Calibrated::repriced(fit, &market, &model, self.options.loss_metrics, true);
    fit.converged = heston_fit.as_ref().is_none_or(|f| f.converged)
      && leverage_finite
      && fit.max_error.is_some_and(f64::is_finite);
    Ok(fit.with_diagnostics(heston_fit))
  }
}
