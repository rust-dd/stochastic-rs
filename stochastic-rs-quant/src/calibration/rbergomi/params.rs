use super::ETA_MIN;
use super::H_MAX;
use super::H_MIN;
use super::RHO_BOUND;
use super::XI0_MIN;
use crate::QuantError;

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum RBergomiXi0 {
  /// $\xi_0(t)=\theta_0$
  Constant(f64),
  /// $\xi_0(t)=\sum_{l=1}^L \theta_l \mathbf{1}_{[T_{l-1},T_l)}(t)$
  PiecewiseConstant {
    maturities: Vec<f64>,
    values: Vec<f64>,
  },
  /// $\xi_0(t)=\beta_0+\beta_1e^{-t/\tau}+\beta_2(t/\tau)e^{-t/\tau}$
  NelsonSiegel {
    beta0: f64,
    beta1: f64,
    beta2: f64,
    tau: f64,
  },
}

impl RBergomiXi0 {
  pub fn value(&self, t: f64) -> f64 {
    match self {
      Self::Constant(level) => level.max(XI0_MIN),
      Self::PiecewiseConstant { maturities, values } => {
        if maturities.len() < 2 || values.is_empty() {
          return XI0_MIN;
        }
        if t < maturities[0] {
          return values[0].max(XI0_MIN);
        }
        for i in 1..maturities.len() {
          if t < maturities[i] {
            return values[i - 1].max(XI0_MIN);
          }
        }
        values[values.len() - 1].max(XI0_MIN)
      }
      Self::NelsonSiegel {
        beta0,
        beta1,
        beta2,
        tau,
      } => {
        let tau = tau.abs().max(1e-6);
        let x = (t.max(0.0)) / tau;
        (beta0 + beta1 * (-x).exp() + beta2 * x * (-x).exp()).max(XI0_MIN)
      }
    }
  }

  pub(super) fn project_in_place(&mut self) {
    match self {
      Self::Constant(level) => {
        *level = level.abs().max(XI0_MIN);
      }
      Self::PiecewiseConstant { values, .. } => {
        for v in values.iter_mut() {
          *v = v.abs().max(XI0_MIN);
        }
      }
      Self::NelsonSiegel { tau, .. } => {
        *tau = tau.abs().max(1e-6);
      }
    }
  }

  pub(super) fn flattened_len(&self) -> usize {
    match self {
      Self::Constant(_) => 1,
      Self::PiecewiseConstant { values, .. } => values.len(),
      Self::NelsonSiegel { .. } => 4,
    }
  }

  pub(super) fn flatten_into(&self, out: &mut Vec<f64>) {
    match self {
      Self::Constant(level) => out.push(*level),
      Self::PiecewiseConstant { values, .. } => out.extend(values.iter().copied()),
      Self::NelsonSiegel {
        beta0,
        beta1,
        beta2,
        tau,
      } => {
        out.push(*beta0);
        out.push(*beta1);
        out.push(*beta2);
        out.push(*tau);
      }
    }
  }

  pub(super) fn assign_from_flattened(&mut self, values: &[f64], offset: &mut usize) {
    match self {
      Self::Constant(level) => {
        *level = values[*offset];
        *offset += 1;
      }
      Self::PiecewiseConstant { values: out, .. } => {
        let end = *offset + out.len();
        out.copy_from_slice(&values[*offset..end]);
        *offset = end;
      }
      Self::NelsonSiegel {
        beta0,
        beta1,
        beta2,
        tau,
      } => {
        *beta0 = values[*offset];
        *beta1 = values[*offset + 1];
        *beta2 = values[*offset + 2];
        *tau = values[*offset + 3];
        *offset += 4;
      }
    }
  }

  pub(super) fn validate(&self) -> Result<(), QuantError> {
    match self {
      Self::Constant(level) => {
        if !level.is_finite() || *level <= 0.0 {
          return Err(QuantError::invalid(
            "RBergomiXi0::Constant",
            format!("must be finite and positive, got {level}"),
          ));
        }
      }
      Self::PiecewiseConstant { maturities, values } => {
        const WHAT: &str = "RBergomiXi0::PiecewiseConstant";
        if maturities.len() < 2 {
          return Err(QuantError::invalid(
            WHAT,
            format!(
              "requires at least two maturity pillars, got {}",
              maturities.len()
            ),
          ));
        }
        if values.len() + 1 != maturities.len() {
          return Err(QuantError::DimensionMismatch {
            what: "RBergomiXi0::PiecewiseConstant values, one per maturity interval",
            expected: maturities.len() - 1,
            got: values.len(),
          });
        }
        if !maturities.windows(2).all(|w| w[0] < w[1]) {
          return Err(QuantError::invalid(
            WHAT,
            format!("maturities must be strictly increasing, got {maturities:?}"),
          ));
        }
        if let Some((i, v)) = values
          .iter()
          .enumerate()
          .find(|(_, v)| !v.is_finite() || **v <= 0.0)
        {
          return Err(QuantError::invalid(
            WHAT,
            format!("value {i} must be finite and positive, got {v}"),
          ));
        }
      }
      Self::NelsonSiegel {
        beta0,
        beta1,
        beta2,
        tau,
      } => {
        if !beta0.is_finite() || !beta1.is_finite() || !beta2.is_finite() || !tau.is_finite() {
          return Err(QuantError::invalid(
            "RBergomiXi0::NelsonSiegel",
            format!(
              "parameters must be finite, got beta0 = {beta0}, beta1 = {beta1}, beta2 = {beta2}, \
               tau = {tau}"
            ),
          ));
        }
        if *tau <= 0.0 {
          return Err(QuantError::invalid(
            "RBergomiXi0::NelsonSiegel",
            format!("tau must be positive, got {tau}"),
          ));
        }
      }
    }
    Ok(())
  }
}

#[derive(Clone, Debug)]
pub struct RBergomiParams {
  /// Hurst exponent in $(0,\frac12)$ for rough volatility.
  pub hurst: f64,
  /// Instantaneous leverage correlation.
  pub rho: f64,
  /// Vol-of-vol parameter.
  pub eta: f64,
  /// Forward variance curve parameterization.
  pub xi0: RBergomiXi0,
}

impl RBergomiParams {
  /// A start for the projected Adam fit from a fractional-OU fit: the Hurst exponent transfers,
  /// `eta` is `|sigma|`, `xi0` a flat `|mu|` (a long-run variance proxy) and `rho` is `-0.7`.
  pub fn seed_from_fou(fou: stochastic_rs_stats::fou::FouFit) -> Self {
    Self {
      hurst: fou.hurst.clamp(H_MIN, H_MAX),
      rho: -0.7,
      eta: fou.sigma.abs().max(ETA_MIN),
      xi0: RBergomiXi0::Constant(fou.mu.abs().max(1e-4)),
    }
  }

  pub fn project_in_place(&mut self) {
    self.hurst = self.hurst.clamp(H_MIN, H_MAX);
    self.rho = self.rho.clamp(-RHO_BOUND, RHO_BOUND);
    self.eta = self.eta.abs().max(ETA_MIN);
    self.xi0.project_in_place();
  }

  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }

  pub fn flattened_len(&self) -> usize {
    3 + self.xi0.flattened_len()
  }

  pub fn flatten(&self) -> Vec<f64> {
    let mut out = Vec::with_capacity(self.flattened_len());
    out.push(self.hurst);
    out.push(self.rho);
    out.push(self.eta);
    self.xi0.flatten_into(&mut out);
    out
  }

  pub fn assign_flattened(&mut self, values: &[f64]) {
    assert_eq!(
      values.len(),
      self.flattened_len(),
      "Flattened parameter vector length mismatch"
    );
    self.hurst = values[0];
    self.rho = values[1];
    self.eta = values[2];
    let mut offset = 3usize;
    self.xi0.assign_from_flattened(values, &mut offset);
  }

  pub(super) fn validate(&self) -> Result<(), QuantError> {
    if !self.hurst.is_finite() || self.hurst <= 0.0 || self.hurst >= 0.5 {
      return Err(QuantError::invalid(
        "RBergomiParams.hurst",
        format!("must be finite and in (0, 0.5), got {}", self.hurst),
      ));
    }
    if !self.rho.is_finite() || self.rho.abs() > 1.0 {
      return Err(QuantError::invalid(
        "RBergomiParams.rho",
        format!("must be finite and in [-1, 1], got {}", self.rho),
      ));
    }
    if !self.eta.is_finite() || self.eta <= 0.0 {
      return Err(QuantError::invalid(
        "RBergomiParams.eta",
        format!("must be finite and positive, got {}", self.eta),
      ));
    }
    self.xi0.validate()
  }
}

#[derive(Clone, Debug)]
pub struct RBergomiMarketSlice {
  /// Time to maturity in years.
  pub tau: f64,
  /// Market terminal samples $\{S_{T_j}^{\mathrm{MKT},(m)}\}$.
  pub terminal_samples: Vec<f64>,
}

impl RBergomiMarketSlice {
  pub(super) fn validate(&self, index: usize) -> Result<(), QuantError> {
    if !self.tau.is_finite() || self.tau <= 0.0 {
      return Err(QuantError::invalid(
        "RBergomiMarketSlice.tau",
        format!(
          "slice {index}: must be finite and positive, got {}",
          self.tau
        ),
      ));
    }
    if self.terminal_samples.is_empty() {
      return Err(QuantError::invalid(
        "RBergomiMarketSlice.terminal_samples",
        format!("slice {index}: cannot be empty"),
      ));
    }
    if let Some((m, x)) = self
      .terminal_samples
      .iter()
      .enumerate()
      .find(|(_, x)| !x.is_finite())
    {
      return Err(QuantError::invalid(
        "RBergomiMarketSlice.terminal_samples",
        format!("slice {index}: sample {m} must be finite, got {x}"),
      ));
    }
    Ok(())
  }
}

#[derive(Clone, Debug)]
pub struct RBergomiCalibrationConfig {
  /// Monte Carlo path count per maturity in each objective evaluation.
  pub paths: usize,
  /// Time discretization granularity, total steps = ceil(T * steps_per_year).
  pub steps_per_year: usize,
  /// Number of exponentials in mSOE kernel approximation.
  pub msoe_terms: usize,
  /// Base learning rate for projected Adam.
  pub learning_rate: f64,
  /// Relative finite-difference bump for numeric gradient.
  pub finite_diff_eps: f64,
  /// Adam $\beta_1$.
  pub adam_beta1: f64,
  /// Adam $\beta_2$.
  pub adam_beta2: f64,
  /// Adam numerical epsilon.
  pub adam_eps: f64,
  /// Optional target tolerance (e.g. bid-ask derived) for early stop.
  pub stop_loss: Option<f64>,
  /// Stop if absolute loss improvement drops below this threshold.
  pub improvement_tol: f64,
}

impl Default for RBergomiCalibrationConfig {
  fn default() -> Self {
    Self {
      paths: 1_024,
      steps_per_year: 128,
      msoe_terms: 12,
      learning_rate: 0.05,
      finite_diff_eps: 1e-3,
      adam_beta1: 0.9,
      adam_beta2: 0.999,
      adam_eps: 1e-8,
      stop_loss: None,
      improvement_tol: 1e-6,
    }
  }
}
