//! # Synthetic CDO tranche
//!
//! $$
//! \mathrm{ETL}(t; A, D) = \mathbb E\bigl[\min(\max(L_t - A, 0), D - A)\bigr],\qquad
//! p_i(t \mid Y) = \Phi\!\left(\frac{\Phi^{-1}(p_i(t)) - \sqrt{\rho}\,Y}{\sqrt{1-\rho}}\right)
//! $$
//!
//! One-factor Gaussian copula pricing of a tranche `[A, D]` on a pool of
//! names. Conditional on the market factor `Y` the defaults are independent,
//! so the portfolio loss distribution follows from the Andersen–Sidenius–Basu
//! recursion on a loss grid, and the unconditional expected tranche loss is
//! the Gauss–Hermite integral over `Y`. The protection leg pays the increase
//! of the expected tranche loss between premium dates, the premium leg
//! accrues the running spread on the expected outstanding tranche notional
//! with the mid-period convention for defaults inside a period (O'Kane 2008,
//! Ch. 17). The large-homogeneous-pool limit of Vasicek is provided as the
//! `N → ∞` cross-check.
//!
//! References: Andersen, L., Sidenius, J. & Basu, S. (2003), *All your hedges
//! in one basket*, Risk 16(11), 67–72; Laurent, J.-P. & Gregory, J. (2005),
//! *Basket default swaps, CDOs and factor copulas*, Journal of Risk 7(4),
//! 103–122; Vasicek, O. (2002), *Loan portfolio value*, Risk 15(12),
//! 160–162; O'Kane, D. (2008), *Modelling Single-name and Multi-name Credit
//! Derivatives*, Wiley, Ch. 12, 17–18.

use std::num::NonZeroUsize;

use gauss_quad::GaussHermite;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;

use crate::credit::survival_curve::SurvivalCurve;
use crate::curves::DiscountCurve;

/// One name of the reference pool.
#[derive(Clone, Debug)]
pub struct PoolName {
  /// Notional weight in the pool (weights sum to one).
  pub weight: f64,
  /// Recovery rate.
  pub recovery: f64,
  /// Survival curve.
  pub survival: SurvivalCurve<f64>,
}

/// Tranche `[attachment, detachment]` of the pool, as fractions of the pool
/// notional, with a running spread on the premium dates.
#[derive(Clone, Debug)]
pub struct CdoTranche {
  pub attachment: f64,
  pub detachment: f64,
  /// Running spread (decimal).
  pub spread: f64,
  /// Premium payment times in years (increasing, last = maturity).
  pub payment_times: Vec<f64>,
  /// Accrual fraction of each period.
  pub accrual: f64,
  /// Asset correlation `ρ` of the one-factor Gaussian copula.
  pub correlation: f64,
  /// Gauss–Hermite nodes for the factor integral.
  pub quadrature_nodes: usize,
  /// Loss grid buckets per unit of pool notional (the ASB recursion rounds
  /// each name's loss to the grid).
  pub loss_buckets: usize,
}

/// Tranche valuation per unit of pool notional.
#[derive(Clone, Debug, PartialEq)]
pub struct TrancheValuation {
  /// Expected tranche loss at each payment time, per unit of pool notional.
  pub expected_loss: Vec<f64>,
  /// Present value of the protection leg.
  pub protection_leg: f64,
  /// Present value of a running spread of one (the tranche's risky annuity).
  pub risky_annuity: f64,
  /// Premium leg at the contract spread.
  pub premium_leg: f64,
  /// Fair running spread `protection / annuity`.
  pub fair_spread: f64,
  /// Upfront the protection buyer pays for the contract spread
  /// (`protection − premium`).
  pub upfront: f64,
}

impl CdoTranche {
  pub fn new(
    attachment: f64,
    detachment: f64,
    spread: f64,
    payment_times: Vec<f64>,
    accrual: f64,
    correlation: f64,
  ) -> Self {
    assert!(
      (0.0..1.0).contains(&attachment) && attachment < detachment && detachment <= 1.0,
      "need 0 ≤ A < D ≤ 1"
    );
    assert!(
      (0.0..1.0).contains(&correlation),
      "correlation must lie in [0, 1)"
    );
    assert!(
      !payment_times.is_empty()
        && payment_times.windows(2).all(|w| w[0] < w[1])
        && payment_times[0] > 0.0,
      "payment times must be positive and increasing"
    );
    Self {
      attachment,
      detachment,
      spread,
      payment_times,
      accrual,
      correlation,
      quadrature_nodes: 40,
      loss_buckets: 400,
    }
  }

  /// Overrides the quadrature and loss-grid resolutions.
  pub fn with_resolution(mut self, quadrature_nodes: usize, loss_buckets: usize) -> Self {
    assert!(
      quadrature_nodes >= 2 && loss_buckets >= 2,
      "quadrature_nodes must satisfy `quadrature_nodes >= 2 && loss_buckets >= 2`, got quadrature_nodes = {quadrature_nodes:?}, loss_buckets = {loss_buckets:?}"
    );
    self.quadrature_nodes = quadrature_nodes;
    self.loss_buckets = loss_buckets;
    self
  }

  /// Expected tranche loss at `t` per unit of pool notional.
  pub fn expected_tranche_loss(&self, pool: &[PoolName], t: f64) -> f64 {
    let distribution = self.loss_distribution(pool, t);
    let unit = 1.0 / self.loss_buckets as f64;
    distribution
      .iter()
      .enumerate()
      .map(|(k, p)| p * tranche_payoff(k as f64 * unit, self.attachment, self.detachment))
      .sum()
  }

  /// Unconditional loss distribution on the grid `k / loss_buckets`,
  /// `k = 0..=loss_buckets`, at time `t`.
  pub fn loss_distribution(&self, pool: &[PoolName], t: f64) -> Vec<f64> {
    assert!(!pool.is_empty(), "the pool needs at least one name");
    let buckets = self.loss_buckets;
    let default_probabilities: Vec<f64> = pool
      .iter()
      .map(|n| n.survival.default_probability(t))
      .collect();
    // Each name's loss in grid units, split between the two neighbouring
    // grid points with the probabilities that preserve its expected loss.
    let loss_units: Vec<(usize, f64)> = pool
      .iter()
      .map(|n| {
        let units = n.weight * (1.0 - n.recovery) * buckets as f64;
        let whole = units.floor();
        (whole as usize, units - whole)
      })
      .collect();
    let thresholds: Vec<f64> = default_probabilities
      .iter()
      .map(|p| ndtri(p.clamp(1e-300, 1.0 - 1e-16)))
      .collect();
    let quad =
      GaussHermite::new(NonZeroUsize::new(self.quadrature_nodes).expect("non-zero node count"));
    let sqrt_rho = self.correlation.sqrt();
    let sqrt_one_minus = (1.0 - self.correlation).sqrt();
    let mut distribution = vec![0.0; buckets + 1];
    // E[g(Y)] for Y ~ N(0, 1) as (1/√π) Σ w_i g(√2 x_i).
    let mut weights_sum = 0.0;
    let nodes_and_weights = quad.nodes().zip(quad.weights());
    let mut scratch = vec![0.0; buckets + 1];
    for (x, w) in nodes_and_weights {
      let (x, w) = (*x, *w);
      let y = std::f64::consts::SQRT_2 * x;
      let conditional = asb_recursion(
        &thresholds
          .iter()
          .map(|&c| norm_cdf((c - sqrt_rho * y) / sqrt_one_minus))
          .collect::<Vec<f64>>(),
        &loss_units,
        buckets,
      );
      for (acc, p) in scratch.iter_mut().zip(&conditional) {
        *acc += w * p;
      }
      weights_sum += w;
    }
    for (acc, s) in distribution.iter_mut().zip(&scratch) {
      *acc = s / weights_sum;
    }
    distribution
  }

  /// Full valuation on `pool` with `discount`.
  ///
  /// The fair spread is NaN when the risky annuity is not positive, as it is
  /// once the expected loss has consumed the tranche width.
  pub fn valuation(&self, pool: &[PoolName], discount: &DiscountCurve<f64>) -> TrancheValuation {
    let width = self.detachment - self.attachment;
    let expected_loss: Vec<f64> = self
      .payment_times
      .iter()
      .map(|&t| self.expected_tranche_loss(pool, t))
      .collect();
    let mut protection = 0.0;
    let mut annuity = 0.0;
    let mut prev_loss = 0.0;
    for (&t, &loss) in self.payment_times.iter().zip(&expected_loss) {
      let df = discount.discount_factor(t);
      protection += df * (loss - prev_loss);
      annuity += df * self.accrual * (width - 0.5 * (loss + prev_loss));
      prev_loss = loss;
    }
    let fair_spread = if annuity > 0.0 {
      protection / annuity
    } else {
      f64::NAN
    };
    TrancheValuation {
      expected_loss,
      protection_leg: protection,
      risky_annuity: annuity,
      premium_leg: self.spread * annuity,
      fair_spread,
      upfront: protection - self.spread * annuity,
    }
  }

  /// Vasicek large-homogeneous-pool limit of the expected tranche loss for a
  /// pool with common default probability `p` and loss-given-default `lgd`:
  /// `E[min(max(lgd·Φ((Φ⁻¹(p) − √ρ Y)/√(1−ρ)) − A, 0), D − A)]`.
  pub fn large_pool_expected_tranche_loss(&self, p: f64, lgd: f64) -> f64 {
    let quad =
      GaussHermite::new(NonZeroUsize::new(self.quadrature_nodes).expect("non-zero node count"));
    let c = ndtri(p.clamp(1e-300, 1.0 - 1e-16));
    let sqrt_rho = self.correlation.sqrt();
    let sqrt_one_minus = (1.0 - self.correlation).sqrt();
    let mut acc = 0.0;
    let mut weights_sum = 0.0;
    for (x, w) in quad.nodes().zip(quad.weights()) {
      let (x, w) = (*x, *w);
      let y = std::f64::consts::SQRT_2 * x;
      let loss = lgd * norm_cdf((c - sqrt_rho * y) / sqrt_one_minus);
      acc += w * tranche_payoff(loss, self.attachment, self.detachment);
      weights_sum += w;
    }
    acc / weights_sum
  }
}

/// `min(max(loss − A, 0), D − A)`.
fn tranche_payoff(loss: f64, attachment: f64, detachment: f64) -> f64 {
  (loss - attachment).max(0.0).min(detachment - attachment)
}

/// Andersen–Sidenius–Basu recursion: distribution of the pool loss on the
/// grid `0..=buckets` given independent defaults with probabilities `p_i`.
/// A name's loss of `k + f` grid units is booked as `k + 1` units with
/// probability `f` and `k` units otherwise, which keeps the expected loss
/// exact on any grid instead of rounding it.
fn asb_recursion(conditional_pd: &[f64], loss_units: &[(usize, f64)], buckets: usize) -> Vec<f64> {
  let mut dist = vec![0.0; buckets + 1];
  dist[0] = 1.0;
  let mut max_loss = 0usize;
  for (&p, &(units, spread)) in conditional_pd.iter().zip(loss_units) {
    if units == 0 && spread == 0.0 {
      continue;
    }
    let extra = usize::from(spread > 0.0);
    let upper = (max_loss + units + extra).min(buckets);
    let p_low = p * (1.0 - spread);
    let p_high = p * spread;
    for k in (0..=upper).rev() {
      let mut value = dist[k] * (1.0 - p);
      if k >= units {
        value += dist[k - units] * p_low;
      }
      if extra == 1 && k > units {
        value += dist[k - units - 1] * p_high;
      }
      dist[k] = value;
    }
    max_loss = upper;
  }
  dist
}

#[cfg(test)]
mod tests;
