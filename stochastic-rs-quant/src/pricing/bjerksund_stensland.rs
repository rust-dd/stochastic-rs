//! # Bjerksund-Stensland 2002 American Option Approximation
//!
//! Analytical approximation for American call and put options using a
//! flat early-exercise boundary split at $t_1=\tfrac12(\sqrt5-1)T$:
//!
//! $$
//! C_{\mathrm{am}}=\alpha_2 F^{\beta}
//!   -\alpha_2\,\phi(F,t_1,\beta,I_2,I_2)
//!   +\phi(F,t_1,1,I_2,I_2)
//!   -\phi(F,t_1,1,I_1,I_2)
//!   -X\,\phi(F,t_1,0,I_2,I_2)
//!   +X\,\phi(F,t_1,0,I_1,I_2)
//!   +\cdots
//! $$
//!
//! Put values use the Bjerksund-Stensland symmetry relation.
//!
//! Reference: Bjerksund, P. & Stensland, G. (2002). "Closed Form Valuation
//! of American Options." Discussion paper 2002/09, NHH.
//! <https://www.researchgate.net/publication/228801918>

use owens_t::biv_norm;
use stochastic_rs_distributions::special::norm_cdf;

use crate::traits::ModelPricer;

/// Bjerksund-Stensland 2002 pricer for American options.
///
/// The struct holds **model state only** — the volatility. Spot, strike,
/// rate, dividend yield and maturity are the pricing *query* and travel as
/// arguments to [`ModelPricer::price_call`], so one instance prices a whole
/// strike/maturity grid.
///
/// Falls back to the GBS (European) value when early exercise is never
/// optimal (i.e. when `b >= r` for calls).
///
/// ```
/// use stochastic_rs_quant::pricing::bjerksund_stensland::BjerksundStensland2002Pricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = BjerksundStensland2002Pricer::new(0.35);
/// // A dividend-paying American call is worth at least its European value.
/// let american = model.price_call(42.0, 40.0, 0.04, 0.08, 0.75);
/// assert!(american > 42.0 - 40.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BjerksundStensland2002Pricer {
  /// Volatility
  pub v: f64,
}

impl BjerksundStensland2002Pricer {
  pub const fn new(v: f64) -> Self {
    Self { v }
  }

  /// Cost of carry: b = r - q
  fn b(r: f64, q: f64) -> f64 {
    r - q
  }

  /// European GBS call price (used as lower bound).
  fn gbs_call(&self, fs: f64, x: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = ((fs / x).ln() + (b + 0.5 * v * v) * t) / (v * t.sqrt());
    let d2 = d1 - v * t.sqrt();
    fs * ((b - r) * t).exp() * norm_cdf(d1) - x * (-r * t).exp() * norm_cdf(d2)
  }

  /// European GBS put price.
  fn gbs_put(&self, fs: f64, x: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
    let d1 = ((fs / x).ln() + (b + 0.5 * v * v) * t) / (v * t.sqrt());
    let d2 = d1 - v * t.sqrt();
    x * (-r * t).exp() * norm_cdf(-d2) - fs * ((b - r) * t).exp() * norm_cdf(-d1)
  }

  /// The $\phi$ intermediate function.
  fn phi(&self, fs: f64, t: f64, gamma: f64, h: f64, i: f64, r: f64, b: f64, v: f64) -> f64 {
    let v2 = v * v;
    let d1 = -((fs / h).ln() + (b + (gamma - 0.5) * v2) * t) / (v * t.sqrt());
    let d2 = d1 - 2.0 * (i / fs).ln() / (v * t.sqrt());

    let lambda = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * v2;
    let kappa = 2.0 * b / v2 + (2.0 * gamma - 1.0);

    (lambda * t).exp() * fs.powf(gamma) * (norm_cdf(d1) - (i / fs).powf(kappa) * norm_cdf(d2))
  }

  /// The $\psi$ intermediate function (uses bivariate normal CDF).
  fn psi(
    &self,
    fs: f64,
    t2: f64,
    gamma: f64,
    h: f64,
    i2: f64,
    i1: f64,
    t1: f64,
    r: f64,
    b: f64,
    v: f64,
  ) -> f64 {
    let v2 = v * v;
    let vsqrt_t1 = v * t1.sqrt();
    let vsqrt_t2 = v * t2.sqrt();

    let bgamma_t1 = (b + (gamma - 0.5) * v2) * t1;
    let bgamma_t2 = (b + (gamma - 0.5) * v2) * t2;

    let d1 = ((fs / i1).ln() + bgamma_t1) / vsqrt_t1;
    let d2 = ((i2 * i2 / (fs * i1)).ln() + bgamma_t1) / vsqrt_t1;
    let d3 = ((fs / i1).ln() - bgamma_t1) / vsqrt_t1;
    let d4 = ((i2 * i2 / (fs * i1)).ln() - bgamma_t1) / vsqrt_t1;

    let e1 = ((fs / h).ln() + bgamma_t2) / vsqrt_t2;
    let e2 = ((i2 * i2 / (fs * h)).ln() + bgamma_t2) / vsqrt_t2;
    let e3 = ((i1 * i1 / (fs * h)).ln() + bgamma_t2) / vsqrt_t2;
    let e4 = ((fs * i1 * i1 / (h * i2 * i2)).ln() + bgamma_t2) / vsqrt_t2;

    let tau = (t1 / t2).sqrt();
    let lambda = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * v2;
    let kappa = 2.0 * b / v2 + (2.0 * gamma - 1.0);

    // owens_t::biv_norm computes P(X > x, Y > y), so negate args for CDF
    let cbnd = |a: f64, b: f64, rho: f64| -> f64 { biv_norm(-a, -b, rho) };

    (lambda * t2).exp()
      * fs.powf(gamma)
      * (cbnd(-d1, -e1, tau)
        - (i2 / fs).powf(kappa) * cbnd(-d2, -e2, tau)
        - (i1 / fs).powf(kappa) * cbnd(-d3, -e3, -tau)
        + (i1 / i2).powf(kappa) * cbnd(-d4, -e4, -tau))
  }

  /// Core BS2002 call pricing (works on transformed inputs for puts).
  fn bs2002_call(&self, fs: f64, x: f64, t: f64, r: f64, b: f64, v: f64) -> f64 {
    let e_value = self.gbs_call(fs, x, t, r, b, v);

    // If b >= r, early exercise is never optimal
    if b >= r {
      return e_value;
    }

    let v2 = v * v;
    let t1 = 0.5 * (5.0_f64.sqrt() - 1.0) * t;
    let t2 = t;

    let beta_inside = ((b / v2 - 0.5).powi(2) + 2.0 * r / v2).abs();
    let beta = (0.5 - b / v2) + beta_inside.sqrt();
    let b_infinity = (beta / (beta - 1.0)) * x;
    let b_zero = f64::max(x, (r / (r - b)) * x);

    let h1 = -(b * t1 + 2.0 * v * t1.sqrt()) * (x * x / ((b_infinity - b_zero) * b_zero));
    let h2 = -(b * t2 + 2.0 * v * t2.sqrt()) * (x * x / ((b_infinity - b_zero) * b_zero));

    let i1 = b_zero + (b_infinity - b_zero) * (1.0 - h1.exp());
    let i2 = b_zero + (b_infinity - b_zero) * (1.0 - h2.exp());

    let alpha1 = (i1 - x) * i1.powf(-beta);
    let alpha2 = (i2 - x) * i2.powf(-beta);

    // Check for immediate exercise
    if fs >= i2 {
      return fs - x;
    }

    let value = alpha2 * fs.powf(beta) - alpha2 * self.phi(fs, t1, beta, i2, i2, r, b, v)
      + self.phi(fs, t1, 1.0, i2, i2, r, b, v)
      - self.phi(fs, t1, 1.0, i1, i2, r, b, v)
      - x * self.phi(fs, t1, 0.0, i2, i2, r, b, v)
      + x * self.phi(fs, t1, 0.0, i1, i2, r, b, v)
      + alpha1 * self.phi(fs, t1, beta, i1, i2, r, b, v)
      - alpha1 * self.psi(fs, t2, beta, i1, i2, i1, t1, r, b, v)
      + self.psi(fs, t2, 1.0, i1, i2, i1, t1, r, b, v)
      - self.psi(fs, t2, 1.0, x, i2, i1, t1, r, b, v)
      - x * self.psi(fs, t2, 0.0, i1, i2, i1, t1, r, b, v)
      + x * self.psi(fs, t2, 0.0, x, i2, i1, t1, r, b, v);

    // Ensure at least the European value
    f64::max(value, e_value)
  }

  /// American call and put price at one query point.
  pub fn call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    let b = Self::b(r, q);

    // Call: direct BS2002
    let call = self.bs2002_call(s, k, tau, r, b, self.v);

    // Put: use the Bjerksund-Stensland symmetry relation
    // P(S, X, T, r, b, v) = C(X, S, T, r-b, -b, v)
    let put_as_call = self.bs2002_call(k, s, tau, r - b, -b, self.v);
    let put = f64::max(put_as_call, self.gbs_put(s, k, tau, r, b, self.v));

    (call, put)
  }
}

impl ModelPricer for BjerksundStensland2002Pricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).0
  }

  /// Overrides the trait's vanilla-parity default. Put-call parity holds
  /// for European options only; an American put carries an early-exercise
  /// premium the call does not, so the default would understate it. This
  /// returns the Bjerksund-Stensland symmetry value
  /// $P(S,X,T,r,b,v)=C(X,S,T,r-b,-b,v)$, floored at the European put — see
  /// `bs2002_price_put_overrides_vanilla_parity`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).1
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  const S: f64 = 100.0;
  const K: f64 = 105.0;
  const R: f64 = 0.05;
  const Q: f64 = 0.02;
  const TAU: f64 = 0.75;
  const V: f64 = 0.25;

  /// Cross-arch tolerance: the goldens route through `norm_cdf` and
  /// `biv_norm`, whose last bits differ between aarch64-darwin and CI's
  /// ubuntu x86_64 under FMA contraction.
  const TOL: f64 = 1e-12;

  /// Captured from `PricerExt::calculate_call_put()` **before** the
  /// `ModelPricer` reshape. The reshape is an API change only.
  #[test]
  fn bs2002_model_pricer_matches_pre_refactor_goldens() {
    let model = BjerksundStensland2002Pricer::new(V);
    let (call, put) = model.call_put(S, K, R, Q, TAU);
    assert!((call - 7.356284498106589).abs() < TOL, "call {call}");
    assert!((put - 10.292920281301193).abs() < TOL, "put {put}");
    assert_eq!(model.price_call(S, K, R, Q, TAU), call);
    assert_eq!(model.price_put(S, K, R, Q, TAU), put);
  }

  /// American puts carry an early-exercise premium the call does not, so
  /// European put-call parity — the trait's `price_put` default — is not
  /// merely imprecise here, it is the wrong model.
  #[test]
  fn bs2002_price_put_overrides_vanilla_parity() {
    let model = BjerksundStensland2002Pricer::new(V);
    let (call, put) = model.call_put(S, K, R, Q, TAU);
    let vanilla = call - S * (-Q * TAU).exp() + K * (-R * TAU).exp();
    assert!(
      put > vanilla + 1e-3,
      "American put must exceed the European-parity value: {put} vs {vanilla}"
    );
  }

  /// The capability the reshape exists for: one model, a whole grid.
  #[test]
  fn bs2002_one_model_prices_a_grid() {
    let model = BjerksundStensland2002Pricer::new(V);
    for &tau in &[0.25, 0.5, 1.0] {
      let mut prev = f64::INFINITY;
      for &k in &[90.0, 100.0, 110.0] {
        let c = model.price_call(S, k, R, Q, tau);
        assert!(c.is_finite() && c < prev, "call must fall in strike");
        prev = c;
      }
    }
  }
}
