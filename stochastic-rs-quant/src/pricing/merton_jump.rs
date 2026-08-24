//! # Merton Jump
//!
//! $$
//! V=\sum_{n=0}^{\infty}e^{-\lambda T}\frac{(\lambda T)^n}{n!}V_{BS}(\sigma_n,r_n)
//! $$
//!
use super::bsm::BSMCoc;
use super::bsm::BSMPricer;
use crate::traits::ModelPricer;

mod greeks;

/// Merton (1976) jump-diffusion pricer.
///
/// The struct holds **model state only** — the total volatility, the jump
/// intensity and jump-variance share, the Poisson-series truncation limit
/// and the cost-of-carry convention. Spot, strike, rate, dividend yield and
/// maturity are the pricing *query* and travel as arguments to
/// [`ModelPricer::price_call`] and to every Greek, so one instance prices a
/// whole strike/maturity grid.
///
/// The query's `(r, q)` pair is **`(discount rate, carry offset)`**, not
/// necessarily `(risk-free rate, dividend yield)`: this pricer discounts at
/// `r` and carries at [`BSMPricer::b(r, q)`](BSMPricer::b), whose value
/// depends on `self.b`. To reproduce a discount `r₀` with a carry `b₀` —
/// Garman-Kohlhagen's `b₀ = r_d − r_f`, say — pass
/// `(r, q) = (r₀, r₀ − b₀)`, which is an identity rather than an
/// approximation because GK's `b(r, q) = r − q`. See
/// `merton_gk_carries_at_rd_minus_rf_and_discounts_at_r`.
///
/// ```
/// use stochastic_rs_quant::pricing::bsm::BSMCoc;
/// use stochastic_rs_quant::pricing::merton_jump::Merton1976Pricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = Merton1976Pricer::new(0.2, 0.5, 0.4, 10, BSMCoc::Bsm1973);
/// let atm = model.price_call(100.0, 100.0, 0.05, 0.0, 0.5);
/// let otm = model.price_call(100.0, 120.0, 0.05, 0.0, 0.5);
/// assert!(atm > otm);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Merton1976Pricer {
  /// Volatility
  pub v: f64,
  /// Expected number of jumps
  pub lambda: f64,
  /// Percentage of the volatility due to jumps
  pub gamma: f64,
  /// Iteration limit
  pub m: usize,
  /// Cost of carry
  pub b: BSMCoc,
}

impl Merton1976Pricer {
  pub const fn new(v: f64, lambda: f64, gamma: f64, m: usize, b: BSMCoc) -> Self {
    Self {
      v,
      lambda,
      gamma,
      m,
      b,
    }
  }

  /// Poisson-weighted series $\sum_{n=0}^{m-1} w_n \cdot V_{BS}(\sigma_n)$,
  /// routed through [`poisson_weight`](Self::poisson_weight)'s
  /// running-product weight rather than an integer `n!` — the latter
  /// overflows `usize` past `n \approx 21` (the crate's Python binding
  /// documents a default of `m = 50`), silently producing garbage instead
  /// of a price. Numerically identical to the pre-refactor factorial-based
  /// loop for `m \le 20` (see
  /// `merton_price_m10_matches_pre_refactor_value` for the regression pin).
  pub fn call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    let mut call = 0.0;
    let mut put = 0.0;

    for i in 0..self.m {
      let weight = self.poisson_weight(i, tau);
      let (c, p) = self.term_bsm(i, tau).call_put(s, k, r, q, tau);
      call += c * weight;
      put += p * weight;
    }

    (call, put)
  }

  /// Jump-size standard deviation implied by decomposing total volatility
  /// `v` into a diffusive part and a jump part that together explain a
  /// `gamma` fraction of the variance.
  fn jump_size_std(&self) -> f64 {
    (self.v.powi(2) * self.gamma / self.lambda).sqrt()
  }

  /// Diffusive volatility component (total variance minus the jump
  /// contribution).
  fn diffusive_std(&self) -> f64 {
    (self.v.powi(2) - self.lambda * self.jump_size_std().powi(2)).sqrt()
  }

  /// Per-term volatility used by the `n`-th element of the Poisson-weighted
  /// series, so Greeks built from it stay exact derivatives of the price
  /// the pricer actually returns.
  fn term_vol(&self, n: usize, tau: f64) -> f64 {
    ((self.diffusive_std().powi(2) + self.jump_size_std().powi(2)) * n as f64 / tau).sqrt()
  }

  /// Poisson weight `e^{-λτ}(λτ)^n / n!` for the `n`-th term. Accumulates
  /// `(λτ)^n / n!` as a running `f64` product rather than an integer `n!`
  /// (which overflows `usize` past `n ≈ 20`, unlike the Poisson weight
  /// itself — bounded in `[0, 1]` for every `n`).
  fn poisson_weight(&self, n: usize, tau: f64) -> f64 {
    let lt = self.lambda * tau;
    let ratio = (0..n).fold(1.0, |acc, i| acc * lt / (i as f64 + 1.0));
    (-lt).exp() * ratio
  }

  /// `BSMPricer` at this pricer's cost-of-carry convention and total
  /// volatility `self.v` (the no-jump / Black-Scholes limit);
  /// [`term_bsm`](Self::term_bsm) swaps in the per-term volatility.
  fn base_bsm(&self) -> BSMPricer {
    BSMPricer::new(self.v, self.b)
  }

  fn term_bsm(&self, n: usize, tau: f64) -> BSMPricer {
    BSMPricer::new(self.term_vol(n, tau), self.b)
  }
}

impl ModelPricer for Merton1976Pricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).0
  }

  /// Overrides the trait's vanilla-parity default for the same reason
  /// [`BSMPricer::price_put`] does, inherited term by term: each element of
  /// the Poisson series carries at $e^{(b-r)\tau}$, which equals the
  /// default's $e^{-q\tau}$ only when $b = r - q$ — false for
  /// [`BSMCoc::Bsm1973`] at `q != 0` and for [`BSMCoc::Black1976`] /
  /// [`BSMCoc::Asay1982`]. See `merton_price_put_overrides_vanilla_parity`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).1
  }
}

#[cfg(test)]
mod tests;
