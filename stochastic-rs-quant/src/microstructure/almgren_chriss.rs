//! Almgren–Chriss execution of $X$ shares over $N$ intervals of $\tau = T/N$ under linear impact,
//! minimising $E\left[C\right] + \lambda\,\mathrm{Var}\left[C\right]$; $\lambda = 0$ is TWAP.
//!
//! $$
//! x_k = X\,\frac{\sinh\bigl(\kappa(T-t_k)\bigr)}{\sinh(\kappa T)},
//! \qquad
//! \kappa = \frac{1}{\tau}\,\mathrm{arccosh}\!\left(
//!   1+\tfrac{\lambda\sigma^2\tau^2}{2\tilde\eta}\right),
//! \qquad
//! \tilde\eta = \eta - \gamma\tau/2.
//! $$
//!
//! Reference: Almgren & Chriss (2001), J. Risk 3(2), 5–39, DOI 10.21314/JOR.2001.041

use ndarray::Array1;

use crate::order_book::Side;
use crate::traits::RealExt;

/// Almgren-Chriss model inputs.
#[derive(Debug, Clone)]
pub struct AlmgrenChrissParams<T: RealExt> {
  /// Whether the parent order sells or buys `total_shares`.
  pub side: Side,
  /// Number of shares to execute (always non-negative).
  pub total_shares: T,
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

impl<T: RealExt> AlmgrenChrissParams<T> {
  /// Inputs with no half-spread cost (`epsilon = 0`).
  pub fn new(
    side: Side,
    total_shares: T,
    horizon: T,
    n_intervals: usize,
    volatility: T,
    gamma: T,
    eta: T,
    lambda: T,
  ) -> Self {
    Self {
      side,
      total_shares,
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

/// Result of an Almgren-Chriss schedule; a buy's three series are a sell's sign reversed
/// (Almgren–Chriss 2001, §2), so its inventory runs from $-X$ up to $0$.
#[derive(Debug, Clone)]
pub struct AlmgrenChrissPlan<T: RealExt> {
  /// Inventory holdings $x_k$ at the end of each interval. For Sell:
  /// $x_0 = +X, \dots, x_N = 0$. For Buy: $x_0 = -X, \dots, x_N = 0$.
  pub inventory: Array1<T>,
  /// Trade sizes $n_k = x_{k-1} - x_k$ executed at each step. Positive for
  /// Sell (shares sold), negative for Buy (shares acquired).
  pub trades: Array1<T>,
  /// Trade rates $v_k = n_k / \tau$. Same sign as `trades`.
  pub rates: Array1<T>,
  /// Decay rate $\kappa$ (zero when `lambda == 0`).
  pub kappa: T,
  /// Modified temporary-impact coefficient $\tilde\eta = \eta - \gamma\tau/2$.
  pub eta_tilde: T,
  /// Expected execution cost $E\left[C\right]$.
  pub expected_cost: T,
  /// Variance of execution cost $\mathrm{Var}\left[C\right]$.
  pub variance: T,
}

impl<T: RealExt> AlmgrenChrissPlan<T> {
  /// Implementation shortfall: $E\left[C\right] + \lambda\,\mathrm{Var}\left[C\right]$.
  pub fn risk_adjusted_cost(&self, lambda: T) -> T {
    self.expected_cost + lambda * self.variance
  }
}

/// Compute the Almgren-Chriss optimal execution schedule.
pub fn optimal_execution<T: RealExt>(params: &AlmgrenChrissParams<T>) -> AlmgrenChrissPlan<T> {
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

  if params.side == Side::Buy {
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
fn sinh<T: RealExt>(x: T) -> T {
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
    let p = AlmgrenChrissParams::new(Side::Sell, 1_000.0_f64, 1.0, 10, 0.01, 1e-7, 1e-5, 0.0);
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
      let p = AlmgrenChrissParams::new(Side::Sell, 1_000.0_f64, 1.0, 10, 0.05, 1e-7, 1e-5, lam);
      optimal_execution(&p)
    };
    let cautious = mk(0.01);
    let aggressive = mk(10.0);
    assert!(aggressive.trades[0] > cautious.trades[0]);
    assert!(aggressive.kappa > cautious.kappa);
  }

  #[test]
  fn risk_adjusted_cost_consistent() {
    let p = AlmgrenChrissParams::new(Side::Sell, 1_000.0_f64, 1.0, 20, 0.02, 2e-7, 5e-6, 0.5);
    let plan = optimal_execution(&p);
    let recomputed = plan.expected_cost + p.lambda * plan.variance;
    assert!(approx(plan.risk_adjusted_cost(p.lambda), recomputed, 1e-12));
  }

  #[test]
  fn buy_direction_negates_rates() {
    let mut p = AlmgrenChrissParams::new(Side::Sell, 500.0_f64, 1.0, 5, 0.01, 1e-7, 1e-5, 0.5);
    let sell = optimal_execution(&p);
    p.side = Side::Buy;
    let buy = optimal_execution(&p);
    for k in 0..p.n_intervals {
      assert!(approx(buy.rates[k], -sell.rates[k], 1e-12));
    }
  }

  #[test]
  fn buy_direction_negates_all_three_series() {
    let mut p = AlmgrenChrissParams::new(Side::Sell, 500.0_f64, 1.0, 5, 0.01, 1e-7, 1e-5, 0.5);
    let sell = optimal_execution(&p);
    p.side = Side::Buy;
    let buy = optimal_execution(&p);
    for k in 0..=p.n_intervals {
      assert!(approx(buy.inventory[k], -sell.inventory[k], 1e-12));
    }
    for k in 0..p.n_intervals {
      assert!(approx(buy.trades[k], -sell.trades[k], 1e-12));
      assert!(approx(buy.rates[k], -sell.rates[k], 1e-12));
    }
    // Buy inventory starts at -X (short position to cover) and ends at 0.
    assert!(approx(buy.inventory[0], -500.0, 1e-9));
    assert!(approx(buy.inventory[p.n_intervals], 0.0, 1e-12));
    // Expected cost and variance are sign-insensitive (squared terms).
    assert!(approx(buy.expected_cost, sell.expected_cost, 1e-9));
    assert!(approx(buy.variance, sell.variance, 1e-9));
  }

  #[test]
  fn full_inventory_ends_at_zero() {
    let p = AlmgrenChrissParams::new(Side::Sell, 2_500.0_f64, 2.0, 50, 0.03, 5e-8, 4e-6, 1.0);
    let plan = optimal_execution(&p);
    assert!(approx(plan.inventory[plan.inventory.len() - 1], 0.0, 1e-9));
    let total: f64 = plan.trades.iter().copied().sum();
    assert!(approx(total, 2_500.0, 1e-6));
  }
}
