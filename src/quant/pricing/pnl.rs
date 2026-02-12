//! Profit and Loss utilities for option pricing and risk decomposition

#[derive(Clone, Debug)]
pub struct GreekPnL {
  /// Delta
  pub delta: f64,
  /// Gamma
  pub gamma: f64,
  /// Vega
  pub vega: f64,
  /// Theta
  pub theta: f64,
}

impl GreekPnL {
  pub fn new(delta: f64, gamma: f64, vega: f64, theta: f64) -> Self {
    Self { delta, gamma, vega, theta }
  }

  /// First-order/second-order Greek-based P&L decomposition over a small interval.
  /// ΔV ≈ Δ ⋅ ΔS + 0.5 Γ (ΔS)^2 + Vega ⋅ Δσ + Θ ⋅ Δt
  #[must_use]
  pub fn calculate(&self, d_s: f64, d_sigma: f64, dt: f64) -> f64 {
    self.delta * d_s + 0.5 * self.gamma * d_s * d_s + self.vega * d_sigma + self.theta * dt
  }
}

#[derive(Clone, Debug)]
pub struct DiscretePnL {
  /// Position quantity
  pub quantity: f64,
  /// Initial price
  pub p0: f64,
}

impl DiscretePnL {
  pub fn new(quantity: f64, p0: f64) -> Self {
    Self { quantity, p0 }
  }

  /// Discrete P&L for a position from time 0 to 1: PnL = q * (p1 - p0)
  #[must_use]
  pub fn calculate(&self, p1: f64) -> f64 {
    self.quantity * (p1 - self.p0)
  }
}

#[derive(Clone, Debug)]
pub struct DeltaHedgedPnL {
  /// Theta
  pub theta: f64,
  /// Gamma
  pub gamma: f64,
  /// Volatility
  pub sigma: f64,
  /// Spot price
  pub s: f64,
}

impl DeltaHedgedPnL {
  pub fn new(theta: f64, gamma: f64, sigma: f64, s: f64) -> Self {
    Self { theta, gamma, sigma, s }
  }

  /// Delta-hedged option P&L approximation over dt (ignoring funding/costs):
  /// dΠ ≈ Θ dt + 0.5 Γ σ^2 S^2 dt
  #[must_use]
  pub fn calculate(&self, dt: f64) -> f64 {
    self.theta * dt + 0.5 * self.gamma * self.sigma * self.sigma * self.s * self.s * dt
  }
}
