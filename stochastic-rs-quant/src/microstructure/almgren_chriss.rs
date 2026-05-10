//! Almgren-Chriss optimal execution under linear permanent and temporary
//! price impact.
//!
//! Liquidate $X$ shares (or accumulate, with `ExecutionDirection::Buy`) over
//! $N$ equally spaced trading intervals of length $\tau = T/N$. Permanent
//! impact $g(v) = \gamma v$, linear temporary impact $h(v) = \eta v$ plus a
//! fixed half-spread $\epsilon\,\mathrm{sign}(v)$.
//!
//! Mean-variance objective $E[C] + \lambda\,\mathrm{Var}[C]$ admits the
//! closed-form trajectory
//!
//! $$
//! x_k = X\,\frac{\sinh\bigl(\kappa(T-t_k)\bigr)}{\sinh(\kappa T)},
//! \qquad
//! \kappa = \frac{1}{\tau}\,\mathrm{arccosh}\!\left(1+\tfrac{\lambda\sigma^2\tau^2}{2\tilde\eta}\right),
//! \qquad
//! \tilde\eta = \eta - \gamma\tau/2.
//! $$
//!
//! Setting $\lambda = 0$ recovers the linear (TWAP) inventory schedule;
//! letting $\lambda\to\infty$ liquidates immediately.
//!
//! Reference: Almgren, Chriss, "Optimal Execution of Portfolio Transactions",
//! Journal of Risk, 3(2), 5-39 (2001). DOI: 10.21314/JOR.2001.041

use std::fmt::Display;

use ndarray::Array1;

use crate::traits::FloatExt;

/// Direction of the parent order.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionDirection {
  /// Sell `total_shares` over `[0, T]`.
  #[default]
  Sell,
  /// Buy `total_shares` over `[0, T]`.
  Buy,
}

impl Display for ExecutionDirection {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Sell => write!(f, "Sell"),
      Self::Buy => write!(f, "Buy"),
    }
  }
}

/// Almgren-Chriss model inputs.
#[derive(Debug, Clone)]
pub struct AlmgrenChrissParams<T: FloatExt> {
  /// Number of shares to execute (always non-negative).
  pub total_shares: T,
  /// Direction of the parent order.
  pub direction: ExecutionDirection,
  /// Total trading horizon (e.g. seconds, days, fractions of a day).
  pub horizon: T,
  /// Number of equally spaced trading intervals.
  pub n_intervals: usize,
  /// Mid-price volatility (in the same time units as `horizon`).
  pub volatility: T,
  /// Linear permanent-impact coefficient $\gamma$ (price per share).
  pub gamma: T,
  /// Linear temporary-impact coefficient $\eta$ (price per (share / time)).
  pub eta: T,
  /// Fixed-cost / half-spread component $\epsilon$ paid per share traded.
  pub epsilon: T,
  /// Risk-aversion parameter $\lambda \ge 0$.
  pub lambda: T,
}

impl<T: FloatExt> AlmgrenChrissParams<T> {
  /// Construct with sensible defaults: `epsilon = 0`, `direction = Sell`.
  pub fn new(
    total_shares: T,
    horizon: T,
    n_intervals: usize,
    volatility: T,
    gamma: T,
    eta: T,
    lambda: T,
  ) -> Self {
    Self {
      total_shares,
      direction: ExecutionDirection::Sell,
      horizon,
      n_intervals,
      volatility,
      gamma,
      eta,
      epsilon: T::zero(),
      lambda,
    }
  }
}

/// Result of an Almgren-Chriss schedule computation.
#[derive(Debug, Clone)]
pub struct AlmgrenChrissPlan<T: FloatExt> {
  /// Inventory holdings $x_k$ at the end of each interval, $x_0 = X, \dots, x_N = 0$.
  pub inventory: Array1<T>,
  /// Trade sizes $n_k = x_{k-1} - x_k$ executed at each step (positive = sells).
  pub trades: Array1<T>,
  /// Trade rates $v_k = n_k / \tau$.
  pub rates: Array1<T>,
  /// Decay rate $\kappa$ (zero when `lambda == 0`).
  pub kappa: T,
  /// Modified temporary-impact coefficient $\tilde\eta = \eta - \gamma\tau/2$.
  pub eta_tilde: T,
  /// Expected execution cost $E[C]$.
  pub expected_cost: T,
  /// Variance of execution cost $\mathrm{Var}[C]$.
  pub variance: T,
}

impl<T: FloatExt> AlmgrenChrissPlan<T> {
  /// Implementation shortfall: $E[C] + \lambda\,\mathrm{Var}[C]$.
  pub fn risk_adjusted_cost(&self, lambda: T) -> T {
    self.expected_cost + lambda * self.variance
  }
}

/// Compute the Almgren-Chriss optimal execution schedule.
pub fn optimal_execution<T: FloatExt>(params: &AlmgrenChrissParams<T>) -> AlmgrenChrissPlan<T> {
  let n = params.n_intervals;
  assert!(n >= 1, "need at least one trading interval");
  assert!(params.horizon > T::zero(), "horizon must be positive");
  assert!(params.eta > T::zero(), "eta must be positive");
  assert!(params.lambda >= T::zero(), "lambda must be non-negative");
  assert!(
    params.total_shares >= T::zero(),
    "total_shares must be non-negative"
  );

  let tau = params.horizon / T::from_usize_(n);
  let eta_tilde = params.eta - params.gamma * tau / T::from_f64_fast(2.0);
  assert!(
    eta_tilde > T::zero(),
    "eta_tilde non-positive: increase eta or shrink tau"
  );

  let half = T::from_f64_fast(0.5);
  let arg = T::one()
    + params.lambda * params.volatility * params.volatility * tau * tau
      / (T::from_f64_fast(2.0) * eta_tilde);
  let kappa = if params.lambda > T::zero() {
    let argf = arg.to_f64().unwrap();
    T::from_f64_fast(argf.acosh()) / tau
  } else {
    T::zero()
  };

  let mut inventory = Array1::<T>::zeros(n + 1);
  inventory[0] = params.total_shares;
  if kappa > T::zero() {
    let kt = kappa * params.horizon;
    let sinh_kt = sinh(kt);
    for k in 1..=n {
      let tk = T::from_usize_(k) * tau;
      let frac = sinh(kappa * (params.horizon - tk)) / sinh_kt;
      inventory[k] = params.total_shares * frac;
    }
  } else {
    for k in 0..=n {
      let frac = (T::from_usize_(n - k)) / T::from_usize_(n);
      inventory[k] = params.total_shares * frac;
    }
  }
  inventory[n] = T::zero();

  let mut trades = Array1::<T>::zeros(n);
  let mut rates = Array1::<T>::zeros(n);
  for k in 0..n {
    trades[k] = inventory[k] - inventory[k + 1];
    rates[k] = trades[k] / tau;
  }

  let mut expected_cost = half * params.gamma * params.total_shares * params.total_shares
    + params.epsilon * params.total_shares;
  let mut variance_acc = T::zero();
  for k in 0..n {
    let n_k = trades[k];
    expected_cost += (eta_tilde / tau) * n_k * n_k;
  }
  for k in 0..n {
    variance_acc += inventory[k + 1] * inventory[k + 1];
  }
  let variance = params.volatility * params.volatility * tau * variance_acc;

  // For a Buy execution we flip ALL three series (inventory, trades, rates)
  // so the plan describes the inventory and trades a *buyer* would observe
  // (i.e. inventory grows from 0 to +X over [0, T] and trades are positive
  // additions to the book). The previous rc.0/rc.1 implementation only
  // flipped `rates`, which left `inventory` and `trades` in sell-frame and
  // produced "sell-shaped" numbers for downstream consumers.
  if matches!(params.direction, ExecutionDirection::Buy) {
    for k in 0..=n {
      inventory[k] = -inventory[k];
    }
    for k in 0..n {
      trades[k] = -trades[k];
      rates[k] = -rates[k];
    }
  }

  AlmgrenChrissPlan {
    inventory,
    trades,
    rates,
    kappa,
    eta_tilde,
    expected_cost,
    variance,
  }
}

#[inline]
fn sinh<T: FloatExt>(x: T) -> T {
  T::from_f64_fast(x.to_f64().unwrap().sinh())
}

#[cfg(test)]
mod tests {
  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn lambda_zero_recovers_twap() {
    let p = AlmgrenChrissParams::new(1_000.0_f64, 1.0, 10, 0.01, 1e-7, 1e-5, 0.0);
    let plan = optimal_execution(&p);
    for k in 0..=p.n_intervals {
      let expected = 1_000.0 * (1.0 - k as f64 / 10.0);
      assert!(
        approx(plan.inventory[k], expected, 1e-9),
        "twap mismatch at {k}: {} vs {}",
        plan.inventory[k],
        expected
      );
    }
    for k in 0..p.n_intervals {
      assert!(approx(plan.trades[k], 100.0, 1e-9));
    }
    assert!(plan.kappa.abs() < 1e-12);
  }

  #[test]
  fn larger_lambda_front_loads_execution() {
    let mk = |lam: f64| {
      let p = AlmgrenChrissParams::new(1_000.0_f64, 1.0, 10, 0.05, 1e-7, 1e-5, lam);
      optimal_execution(&p)
    };
    let cautious = mk(0.01);
    let aggressive = mk(10.0);
    assert!(aggressive.trades[0] > cautious.trades[0]);
    assert!(aggressive.kappa > cautious.kappa);
  }

  #[test]
  fn risk_adjusted_cost_consistent() {
    let p = AlmgrenChrissParams::new(1_000.0_f64, 1.0, 20, 0.02, 2e-7, 5e-6, 0.5);
    let plan = optimal_execution(&p);
    let recomputed = plan.expected_cost + p.lambda * plan.variance;
    assert!(approx(plan.risk_adjusted_cost(p.lambda), recomputed, 1e-12));
  }

  #[test]
  fn buy_direction_negates_rates() {
    let mut p = AlmgrenChrissParams::new(500.0_f64, 1.0, 5, 0.01, 1e-7, 1e-5, 0.5);
    let sell = optimal_execution(&p);
    p.direction = ExecutionDirection::Buy;
    let buy = optimal_execution(&p);
    for k in 0..p.n_intervals {
      assert!(approx(buy.rates[k], -sell.rates[k], 1e-12));
    }
  }

  #[test]
  fn full_inventory_ends_at_zero() {
    let p = AlmgrenChrissParams::new(2_500.0_f64, 2.0, 50, 0.03, 5e-8, 4e-6, 1.0);
    let plan = optimal_execution(&p);
    assert!(approx(plan.inventory[plan.inventory.len() - 1], 0.0, 1e-9));
    let total: f64 = plan.trades.iter().copied().sum();
    assert!(approx(total, 2_500.0, 1e-6));
  }
}
