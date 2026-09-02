//! Per-expiry Sabr smile calibration for caplets / swaptions.
//!
//! $$
//! (\hat\alpha,\hat\nu,\hat\rho)=\arg\min_{\alpha,\nu,\rho}
//!   \sum_i w_i\bigl(\sigma_{\mathrm{Hagan}}(K_i,F,\tau;\alpha,\beta,\nu,\rho)
//!   -\sigma_i^{mkt}\bigr)^2
//! $$
//!
//! The Cev exponent $\beta$ is held fixed at a user-supplied value (commonly
//! 0.5 or 1.0). The three free parameters $(\alpha,\nu,\rho)$ are minimized
//! against the supplied per-strike Black-76 implied volatilities by
//! Nelder-Mead.
//!
//! ## Negative and near-zero forwards
//!
//! The underlying Hagan expansion requires a strictly positive forward and
//! strike. EUR, JPY and CHF caplet forwards were routinely negative from
//! roughly 2015 to 2022, so [`SabrCapletCalibrator`] carries a `shift`
//! displacing both before they reach the expansion — the same
//! $F_s=F+\text{shift}$, $K_s=K+\text{shift}$ convention as
//! [`ShiftedSabrVolatility`](crate::instruments::option::ShiftedSabrVolatility).
//! `shift` defaults to `0.0` (set it with
//! [`SabrCapletCalibrator::with_shift`]), which reproduces the unshifted
//! formula exactly. `calibrate` returns `Err` naming the offending value
//! rather than panicking when the configured shift still leaves the
//! forward or a strike non-positive.
//!
//! Reference: P. S. Hagan, D. Kumar, A. S. Lesniewski, D. E. Woodward,
//! "Managing Smile Risk", Wilmott Magazine (2002). Shift convention:
//! J. Oblój, "Fine-tune your smile: Correction to Hagan et al.", Wilmott
//! Magazine (2008).

use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::core::State;
use argmin::solver::neldermead::NelderMead;

use crate::pricing::sabr::hagan_implied_vol;

/// Calibrated parameter set for a Sabr caplet smile.
///
/// Parallels [`crate::calibration::SabrParams`] but is dedicated to the
/// caplet calibrator so the two pipelines can evolve independently.
#[derive(Debug, Clone, Copy)]
pub struct SabrCapletParams {
  pub alpha: f64,
  pub beta: f64,
  pub nu: f64,
  pub rho: f64,
  /// Displacement the smile was calibrated under; see the module
  /// documentation. `0.0` unless the calibrator used
  /// [`SabrCapletCalibrator::with_shift`]. `alpha`/`nu`/`rho` above are
  /// expressed in *shifted* coordinates whenever this is nonzero.
  pub shift: f64,
}

/// Calibration result for a Sabr caplet smile.
#[derive(Debug, Clone)]
pub struct SabrCapletCalibrationResult {
  /// Sabr level $\alpha$.
  pub alpha: f64,
  /// Sabr Cev exponent $\beta$ (held fixed during calibration).
  pub beta: f64,
  /// Sabr volatility of volatility $\nu$.
  pub nu: f64,
  /// Sabr correlation $\rho$.
  pub rho: f64,
  /// Displacement the smile was calibrated under; see the module
  /// documentation. `0.0` unless the calibrator used
  /// [`SabrCapletCalibrator::with_shift`]. `alpha`/`nu`/`rho` above are
  /// expressed in *shifted* coordinates whenever this is nonzero.
  pub shift: f64,
  /// Root-mean-square vol error across strikes.
  pub rmse: f64,
  /// Residuals $\sigma_{model}-\sigma_{mkt}$ in strike order.
  pub residuals: Vec<f64>,
  /// True when Nelder-Mead reported convergence.
  pub converged: bool,
}

impl SabrCapletCalibrationResult {
  /// Convert to a [`SabrPricer`](crate::pricing::sabr::SabrPricer) for pricing /
  /// vol-surface generation.
  ///
  /// `alpha`/`nu`/`rho` stay in *shifted* coordinates whenever
  /// `self.shift != 0.0`: [`ModelPricer`](crate::traits::ModelPricer)'s
  /// `price_call` forwards `s`/`k` into the Hagan expansion undisplaced, so
  /// pricing with this model requires shifting `s` and `k` by `self.shift`
  /// yourself first — passing the original unshifted (possibly negative)
  /// values reproduces the same panic this shift exists to avoid.
  /// [`Self::to_shifted_volatility`] does the shifting internally and is
  /// the safer conversion whenever `shift` is nonzero.
  pub fn to_model(&self) -> crate::pricing::sabr::SabrPricer {
    crate::pricing::sabr::SabrPricer {
      alpha: self.alpha,
      beta: self.beta,
      nu: self.nu,
      rho: self.rho,
    }
  }

  /// Convert to a
  /// [`ShiftedSabrVolatility`](crate::instruments::option::ShiftedSabrVolatility)
  /// that applies `self.shift` to forward and strike on every
  /// `implied_volatility` call, so callers never shift by hand. This is the
  /// shift-safe counterpart to [`Self::to_model`], and the recommended
  /// conversion whenever the calibrator used
  /// [`SabrCapletCalibrator::with_shift`].
  pub fn to_shifted_volatility(&self) -> crate::instruments::option::ShiftedSabrVolatility<f64> {
    crate::instruments::option::ShiftedSabrVolatility::new(
      self.alpha, self.beta, self.nu, self.rho, self.shift,
    )
  }
}

impl crate::traits::ToModel for SabrCapletCalibrationResult {
  type Model = crate::pricing::sabr::SabrPricer;
  fn to_model(&self, _r: f64, _q: f64) -> Self::Model {
    SabrCapletCalibrationResult::to_model(self)
  }
}

impl crate::traits::CalibrationResult for SabrCapletCalibrationResult {
  type Params = SabrCapletParams;
  fn rmse(&self) -> f64 {
    self.rmse
  }

  fn converged(&self) -> bool {
    self.converged
  }

  fn params(&self) -> Self::Params {
    SabrCapletParams {
      alpha: self.alpha,
      beta: self.beta,
      nu: self.nu,
      rho: self.rho,
      shift: self.shift,
    }
  }
}

impl crate::traits::Calibrator for SabrCapletCalibrator {
  type InitialGuess = (f64, f64, f64);
  type Params = SabrCapletParams;
  type Output = SabrCapletCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    let mut this = self.clone();
    if let Some(g) = initial {
      this.initial_guess = Some(g);
    }
    this.validate_shift()?;
    Ok(this.solve())
  }
}

/// Sabr caplet smile calibrator — fits $(\alpha,\nu,\rho)$ for a single
/// expiry.
#[derive(Debug, Clone)]
pub struct SabrCapletCalibrator {
  /// Forward rate $F$ seen by every strike on this smile.
  pub forward: f64,
  /// Displacement applied to `forward` and every strike before pricing:
  /// $F_s=F+\text{shift}$, $K_s=K+\text{shift}$ (Oblój 2008's shifted-Sabr
  /// convention — see the module documentation and
  /// [`ShiftedSabrVolatility`](crate::instruments::option::ShiftedSabrVolatility)).
  /// `0.0` via [`Self::new`] reproduces the unshifted Hagan formula
  /// exactly; set a positive shift with [`Self::with_shift`] when `forward`
  /// or a strike can be zero or negative (market convention for EUR
  /// caplets has been 2%-3%, i.e. `0.02`-`0.03`).
  pub shift: f64,
  /// Expiry $\tau$ in years.
  pub expiry: f64,
  /// Fixed Cev exponent $\beta$.
  pub beta: f64,
  /// Market strikes.
  pub strikes: Vec<f64>,
  /// Market Black-76 implied vols aligned with `strikes`.
  pub market_vols: Vec<f64>,
  /// Optional per-strike weights.
  pub weights: Option<Vec<f64>>,
  /// Initial guess for $(\alpha,\nu,\rho)$.
  pub initial_guess: Option<(f64, f64, f64)>,
  /// Maximum Nelder-Mead iterations.
  pub max_iters: u64,
  /// Convergence tolerance on the simplex standard deviation.
  pub sd_tolerance: f64,
}

impl SabrCapletCalibrator {
  /// Construct a calibrator with defaults.
  pub fn new(
    forward: f64,
    expiry: f64,
    beta: f64,
    strikes: Vec<f64>,
    market_vols: Vec<f64>,
  ) -> Self {
    assert_eq!(
      strikes.len(),
      market_vols.len(),
      "strikes and market_vols must have equal length"
    );
    Self {
      forward,
      shift: 0.0,
      expiry,
      beta,
      strikes,
      market_vols,
      weights: None,
      initial_guess: None,
      max_iters: 600,
      sd_tolerance: 1e-10,
    }
  }

  /// Displace `forward` and every strike by `shift` before calibration —
  /// see the module documentation. Use a positive shift (market convention
  /// for EUR caplets: `0.02`-`0.03`) whenever `forward` or a strike can be
  /// zero or negative; `calibrate` then returns `Err` naming the offending
  /// value if the shift is still insufficient.
  pub fn with_shift(mut self, shift: f64) -> Self {
    self.shift = shift;
    self
  }

  /// Forward displaced into shifted coordinates: $F+\text{shift}$.
  fn shifted_forward(&self) -> f64 {
    self.forward + self.shift
  }

  /// Checks that `shift` makes the forward and every strike strictly
  /// positive, as the underlying Hagan expansion requires. Called from
  /// [`Calibrator::calibrate`](crate::traits::Calibrator::calibrate) so an
  /// insufficient shift surfaces as `Err`, naming the offending value,
  /// instead of panicking inside the Nelder-Mead cost callback.
  fn validate_shift(&self) -> Result<(), anyhow::Error> {
    let shift = self.shift;
    let forward = self.forward;
    let shifted_forward = self.shifted_forward();
    if shifted_forward <= 0.0 {
      anyhow::bail!(
        "SabrCapletCalibrator: shift {shift} does not make forward {forward} positive \
         (forward + shift = {shifted_forward} <= 0); use a larger shift"
      );
    }
    for (i, &k) in self.strikes.iter().enumerate() {
      let shifted_k = k + shift;
      if shifted_k <= 0.0 {
        anyhow::bail!(
          "SabrCapletCalibrator: shift {shift} does not make strike[{i}] {k} positive \
           (strike + shift = {shifted_k} <= 0); use a larger shift"
        );
      }
    }
    Ok(())
  }

  fn solve(&self) -> SabrCapletCalibrationResult {
    let weights = self
      .weights
      .clone()
      .unwrap_or_else(|| vec![1.0; self.strikes.len()]);

    let (a0, nu0, rho0) = self.initial_guess.unwrap_or_else(|| {
      let atm = self
        .market_vols
        .iter()
        .zip(self.strikes.iter())
        .min_by(|a, b| {
          (a.1 - self.forward)
            .abs()
            .partial_cmp(&(b.1 - self.forward).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(v, _)| *v)
        .unwrap_or(0.2);
      (
        atm * self.shifted_forward().powf(1.0 - self.beta),
        0.4,
        -0.2,
      )
    });

    let problem = SabrCapletCost {
      forward: self.forward,
      shift: self.shift,
      expiry: self.expiry,
      beta: self.beta,
      strikes: self.strikes.clone(),
      market_vols: self.market_vols.clone(),
      weights: weights.clone(),
    };

    let simplex = vec![
      vec![a0, nu0, rho0],
      vec![a0 * 1.3, nu0, rho0],
      vec![a0, nu0 * 1.5, rho0],
      vec![a0, nu0, (rho0 + 0.2).clamp(-0.99, 0.99)],
    ];

    let mut converged = true;
    let best = match NelderMead::new(simplex.clone()).with_sd_tolerance(self.sd_tolerance) {
      Ok(solver) => match Executor::new(problem.clone(), solver)
        .configure(|s| s.max_iters(self.max_iters))
        .run()
      {
        Ok(res) => res
          .state
          .get_best_param()
          .cloned()
          .unwrap_or_else(|| simplex[0].clone()),
        Err(_) => {
          converged = false;
          simplex[0].clone()
        }
      },
      Err(_) => {
        converged = false;
        simplex[0].clone()
      }
    };

    let alpha = best[0].abs().max(1e-8);
    let nu = best[1].abs().max(1e-8);
    let rho = best[2].clamp(-0.9999, 0.9999);

    let shifted_forward = self.shifted_forward();
    let residuals: Vec<f64> = self
      .strikes
      .iter()
      .zip(self.market_vols.iter())
      .map(|(&k, &v_mkt)| {
        let v_model = hagan_implied_vol(
          k + self.shift,
          shifted_forward,
          self.expiry,
          alpha,
          self.beta,
          nu,
          rho,
        );
        v_model - v_mkt
      })
      .collect();
    let ssr: f64 = residuals
      .iter()
      .zip(weights.iter())
      .map(|(r, w)| (w.sqrt() * r).powi(2))
      .sum();
    let rmse = (ssr / residuals.len().max(1) as f64).sqrt();

    SabrCapletCalibrationResult {
      alpha,
      beta: self.beta,
      nu,
      rho,
      shift: self.shift,
      rmse,
      residuals,
      converged,
    }
  }
}

#[derive(Clone)]
struct SabrCapletCost {
  forward: f64,
  shift: f64,
  expiry: f64,
  beta: f64,
  strikes: Vec<f64>,
  market_vols: Vec<f64>,
  weights: Vec<f64>,
}

impl CostFunction for SabrCapletCost {
  type Param = Vec<f64>;
  type Output = f64;

  fn cost(&self, x: &Self::Param) -> Result<f64, argmin::core::Error> {
    let alpha = x[0].abs().max(1e-8);
    let nu = x[1].abs().max(1e-8);
    let rho = x[2].clamp(-0.9999, 0.9999);
    let shifted_forward = self.forward + self.shift;
    let mut sse = 0.0;
    for ((&k, &v_mkt), &w) in self
      .strikes
      .iter()
      .zip(self.market_vols.iter())
      .zip(self.weights.iter())
    {
      let v_model = hagan_implied_vol(
        k + self.shift,
        shifted_forward,
        self.expiry,
        alpha,
        self.beta,
        nu,
        rho,
      );
      let diff = v_model - v_mkt;
      sse += w * diff * diff;
    }
    Ok(sse)
  }
}
