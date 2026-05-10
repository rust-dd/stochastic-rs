//! # Strategies
//!
//! Trading strategies parameterised by a [`Strategy`] trait. A strategy
//! consumes a per-bar [`MarketBar`] and emits a [`StrategyAction`]
//! (target position, optional rebalance cost). A minimal
//! [`Backtest`] driver iterates through a `&[MarketBar]` and returns a
//! [`BacktestResult`] with per-bar PnL, cumulative equity, and a flat
//! position-vs-time vector.
//!
//! $$
//! V_0=\mathbb E^{\mathbb Q}\!\left[e^{-\int_0^T r_tdt}\,\Pi(X_T)\right]
//! $$

pub mod delta_hedge;

/// A single time-bar of market data fed to a [`Strategy`].
#[derive(Debug, Clone, Copy)]
pub struct MarketBar {
  /// Time index (e.g. days since epoch, or fractional year).
  pub t: f64,
  /// Reference asset price (mid / close).
  pub price: f64,
}

/// Action returned by [`Strategy::on_bar`]. `target_position` is the
/// **signed** number of units the strategy wants to hold *after* this
/// bar; the [`Backtest`] driver computes the trade as
/// `target_position - current_position`. `signal` is an optional
/// human-readable tag for telemetry / plotting.
#[derive(Debug, Clone)]
pub struct StrategyAction {
  pub target_position: f64,
  pub signal: Option<String>,
}

/// Trading strategy interface. Implementors maintain whatever internal
/// state they need (rolling stats, prior signals, fitted params) and
/// emit one [`StrategyAction`] per bar.
pub trait Strategy {
  /// Friendly name (telemetry / logging).
  fn name(&self) -> &'static str;
  /// Reset internal state before a fresh backtest run.
  fn reset(&mut self) {}
  /// React to a single market bar; return the desired post-bar
  /// position.
  fn on_bar(&mut self, bar: MarketBar) -> StrategyAction;
}

/// Result of running a [`Backtest`]. `equity` is the running
/// cumulative cash + position×price valuation; `positions` is the
/// signed position held after each bar.
#[derive(Debug, Clone)]
pub struct BacktestResult {
  pub equity: Vec<f64>,
  pub positions: Vec<f64>,
  pub trades: Vec<f64>,
}

/// Minimal backtest driver: walks a strategy across a price series
/// with a flat per-trade cost (`cost_per_unit`, e.g. brokerage + 1bp
/// slippage on the *change* in position). No financing cost, no margin
/// model — kept intentionally small; users wanting realistic execution
/// modelling should layer this on top of `microstructure::almgren_chriss`.
pub struct Backtest {
  pub cost_per_unit: f64,
}

impl Backtest {
  pub fn new(cost_per_unit: f64) -> Self {
    Self { cost_per_unit }
  }

  /// Run `strategy` over `bars`. Initial position is zero; final
  /// position is whatever the last bar requested.
  pub fn run<S: Strategy>(&self, strategy: &mut S, bars: &[MarketBar]) -> BacktestResult {
    strategy.reset();
    let n = bars.len();
    let mut equity = Vec::with_capacity(n);
    let mut positions = Vec::with_capacity(n);
    let mut trades = Vec::with_capacity(n);

    let mut cash = 0.0_f64;
    let mut position = 0.0_f64;
    for &bar in bars.iter() {
      let action = strategy.on_bar(bar);
      let trade = action.target_position - position;
      cash -= trade * bar.price;
      cash -= trade.abs() * self.cost_per_unit;
      position = action.target_position;
      trades.push(trade);
      positions.push(position);
      equity.push(cash + position * bar.price);
    }

    BacktestResult {
      equity,
      positions,
      trades,
    }
  }
}
