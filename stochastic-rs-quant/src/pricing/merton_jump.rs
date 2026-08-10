//! # Merton Jump
//!
//! $$
//! V=\sum_{n=0}^{\infty}e^{-\lambda T}\frac{(\lambda T)^n}{n!}V_{BS}(\sigma_n,r_n)
//! $$
//!
use super::bsm::BSMCoc;
use super::bsm::BSMPricer;
use crate::OptionType;
use crate::traits::GreeksExt;
use crate::traits::PricerExt;
use crate::traits::TimeExt;

#[derive(Debug, Clone)]
pub struct Merton1976Pricer {
  /// Underlying price
  pub s: f64,
  /// Volatility
  pub v: f64,
  /// Strike price
  pub k: f64,
  /// Risk-free rate
  pub r: f64,
  /// Domestic risk-free rate
  pub r_d: Option<f64>,
  /// Foreign risk-free rate
  pub r_f: Option<f64>,
  /// Dividend yield
  pub q: Option<f64>,
  /// Expected number of jumps
  pub lambda: f64,
  /// Percentage of the volatility due to jumps
  pub gamma: f64,
  /// Iteration limit
  pub m: usize,
  /// Time to maturity in years
  pub tau: Option<f64>,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiration: Option<chrono::NaiveDate>,
  /// Option type
  pub option_type: OptionType,
  /// Cost of carry
  pub b: BSMCoc,
}

impl Merton1976Pricer {
  pub fn new(
    s: f64,
    v: f64,
    k: f64,
    r: f64,
    r_d: Option<f64>,
    r_f: Option<f64>,
    q: Option<f64>,
    lambda: f64,
    gamma: f64,
    m: usize,
    tau: Option<f64>,
    eval: Option<chrono::NaiveDate>,
    expiration: Option<chrono::NaiveDate>,
    option_type: OptionType,
    b: BSMCoc,
  ) -> Self {
    Self {
      s,
      v,
      k,
      r,
      r_d,
      r_f,
      q,
      lambda,
      gamma,
      m,
      tau,
      eval,
      expiration,
      option_type,
      b,
    }
  }

  pub fn builder(
    s: f64,
    v: f64,
    k: f64,
    r: f64,
    lambda: f64,
    gamma: f64,
    m: usize,
  ) -> Merton1976PricerBuilder {
    Merton1976PricerBuilder {
      s,
      v,
      k,
      r,
      r_d: None,
      r_f: None,
      q: None,
      lambda,
      gamma,
      m,
      tau: None,
      eval: None,
      expiration: None,
      option_type: OptionType::Call,
      b: BSMCoc::Bsm1973,
    }
  }
}

#[derive(Debug, Clone)]
pub struct Merton1976PricerBuilder {
  s: f64,
  v: f64,
  k: f64,
  r: f64,
  r_d: Option<f64>,
  r_f: Option<f64>,
  q: Option<f64>,
  lambda: f64,
  gamma: f64,
  m: usize,
  tau: Option<f64>,
  eval: Option<chrono::NaiveDate>,
  expiration: Option<chrono::NaiveDate>,
  option_type: OptionType,
  b: BSMCoc,
}

impl Merton1976PricerBuilder {
  pub fn r_d(mut self, r_d: f64) -> Self {
    self.r_d = Some(r_d);
    self
  }
  pub fn r_f(mut self, r_f: f64) -> Self {
    self.r_f = Some(r_f);
    self
  }
  pub fn q(mut self, q: f64) -> Self {
    self.q = Some(q);
    self
  }
  pub fn tau(mut self, tau: f64) -> Self {
    self.tau = Some(tau);
    self
  }
  pub fn eval(mut self, eval: chrono::NaiveDate) -> Self {
    self.eval = Some(eval);
    self
  }
  pub fn expiration(mut self, expiration: chrono::NaiveDate) -> Self {
    self.expiration = Some(expiration);
    self
  }
  pub fn option_type(mut self, option_type: OptionType) -> Self {
    self.option_type = option_type;
    self
  }
  pub fn coc(mut self, b: BSMCoc) -> Self {
    self.b = b;
    self
  }
  pub fn build(self) -> Merton1976Pricer {
    Merton1976Pricer {
      s: self.s,
      v: self.v,
      k: self.k,
      r: self.r,
      r_d: self.r_d,
      r_f: self.r_f,
      q: self.q,
      lambda: self.lambda,
      gamma: self.gamma,
      m: self.m,
      tau: self.tau,
      eval: self.eval,
      expiration: self.expiration,
      option_type: self.option_type,
      b: self.b,
    }
  }
}

impl PricerExt for Merton1976Pricer {
  fn calculate_call_put(&self) -> (f64, f64) {
    let mut bsm = BSMPricer::new(
      self.s,
      self.v,
      self.k,
      self.r,
      self.r_d,
      self.r_f,
      self.q,
      self.tau,
      self.eval,
      self.expiration,
      self.option_type,
      self.b,
    );

    let mut call = 0.0;
    let mut put = 0.0;

    let delta = || -> f64 { (self.v.powi(2) * self.gamma / self.lambda).sqrt() };
    let z = || -> f64 { (self.v.powi(2) - self.lambda * delta().powi(2)).sqrt() };
    let sigma =
      |i: usize, tau: f64| -> f64 { ((z().powi(2) + delta().powi(2)) * i as f64 / tau).sqrt() };
    let tau = self.tau_or_from_dates();

    for i in 0..self.m {
      bsm.v = sigma(i, tau);
      let f: usize = (1..=i).product();
      let num = (-self.lambda * tau).exp() * (self.lambda * tau).powi(i as i32);

      let (c, p) = bsm.calculate_call_put();
      call += c * num / f as f64;
      put += p * num / f as f64;
    }

    (call, put)
  }

  fn calculate_price(&self) -> f64 {
    let (call, put) = self.calculate_call_put();
    match self.option_type {
      OptionType::Call => call,
      OptionType::Put => put,
    }
  }
}

impl TimeExt for Merton1976Pricer {
  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn eval(&self) -> Option<chrono::NaiveDate> {
    self.eval
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    self.expiration
  }
}

impl Merton1976Pricer {
  /// Jump-size standard deviation implied by decomposing total volatility
  /// `v` into a diffusive part and a jump part that together explain a
  /// `gamma` fraction of the variance. Mirrors the private `delta()`
  /// closure inside [`PricerExt::calculate_call_put`].
  fn jump_size_std(&self) -> f64 {
    (self.v.powi(2) * self.gamma / self.lambda).sqrt()
  }

  /// Diffusive volatility component (total variance minus the jump
  /// contribution). Mirrors the private `z()` closure inside
  /// [`PricerExt::calculate_call_put`].
  fn diffusive_std(&self) -> f64 {
    (self.v.powi(2) - self.lambda * self.jump_size_std().powi(2)).sqrt()
  }

  /// Per-term volatility used by the `n`-th element of the Poisson-weighted
  /// series. Mirrors the private `sigma()` closure inside
  /// [`PricerExt::calculate_call_put`], so Greeks built from it stay exact
  /// derivatives of the price the pricer actually returns.
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

  /// `BSMPricer` sharing every Merton field except volatility, which
  /// defaults to `self.v` (the no-jump / Black-Scholes limit); callers
  /// override `.v` per Poisson term.
  fn base_bsm(&self, tau: f64) -> BSMPricer {
    BSMPricer::new(
      self.s,
      self.v,
      self.k,
      self.r,
      self.r_d,
      self.r_f,
      self.q,
      Some(tau),
      None,
      None,
      self.option_type,
      self.b,
    )
  }

  fn term_bsm(&self, n: usize, tau: f64) -> BSMPricer {
    let mut bsm = self.base_bsm(tau);
    bsm.v = self.term_vol(n, tau);
    bsm
  }

  /// Poisson-weighted series over a closed-form BSM Greek. Exact whenever
  /// the Greek's bump variable enters neither [`term_vol`](Self::term_vol)
  /// nor [`poisson_weight`](Self::poisson_weight) — true for spot and
  /// rate, which is why `delta`/`gamma`/`rho` use this path. `λ ≤ 0`
  /// returns the single surviving (`n = 0`, weight 1) term directly,
  /// sidestepping the `0/0` singularity
  /// [`jump_size_std`](Self::jump_size_std) would otherwise hit.
  ///
  /// `n = 0` is always priced at `term_vol(0, τ) = 0` exactly (a property
  /// of the existing price series, not of this method), which sends
  /// `1/v`-shaped closed forms like [`BSMPricer::gamma`] to `0/0`. That
  /// term's true contribution is its `v → 0⁺` limit, which is `0` for any
  /// off-the-money strike (`norm_pdf(d1) → 0` exponentially, beating the
  /// linear `1/v`) — so a `NaN` contribution here is floored to `0` rather
  /// than poisoning the whole sum.
  fn greek_series(&self, greek: impl Fn(&BSMPricer) -> f64) -> f64 {
    let tau = self.tau_or_from_dates();
    if self.lambda <= 0.0 {
      return greek(&self.base_bsm(tau));
    }
    (0..self.m)
      .map(|n| {
        let contribution = self.poisson_weight(n, tau) * greek(&self.term_bsm(n, tau));
        if contribution.is_nan() {
          0.0
        } else {
          contribution
        }
      })
      .sum()
  }

  /// Overflow-safe re-implementation of [`PricerExt::calculate_price`]'s
  /// Poisson sum, built on [`greek_series`](Self::greek_series) instead of
  /// [`PricerExt::calculate_call_put`]'s own loop. Numerically identical to
  /// `calculate_price()` for `m ≤ 20` — both compute `Σ w_n · BS_n(σ_n)`,
  /// just accumulating the same weight `w_n` via a different (equally
  /// valid) route — but unlike `calculate_call_put()`, this never
  /// overflows for larger `m`, since
  /// [`poisson_weight`](Self::poisson_weight) never materializes `n!` as
  /// an integer. Every finite-difference Greek below calls this instead of
  /// `calculate_price()`, so all 9 Greeks stay valid at `m` values
  /// `calculate_call_put()` itself cannot handle (the crate's Python
  /// binding documents a default of `m = 50`, past `calculate_call_put`'s
  /// `m ≈ 21` `usize`-factorial overflow threshold — a pre-existing
  /// limitation of that method, unrelated to `GreeksExt` and out of scope
  /// to fix here).
  fn series_price(&self) -> f64 {
    self.greek_series(|bsm| bsm.calculate_price())
  }

  const H_TAU: f64 = 1e-5;

  fn h_s(&self) -> f64 {
    self.s.abs() * 1e-4
  }

  fn h_v(&self) -> f64 {
    self.v.abs().max(0.01) * 1e-4
  }

  /// Clone with `s`/`v`/`tau` bumped — backs the Greeks a naive Poisson
  /// series would get wrong (see the [`GreeksExt`] impl doc below).
  fn bumped(&self, ds: f64, dv: f64, dtau: f64) -> Self {
    let mut p = self.clone();
    p.s += ds;
    p.v = (p.v + dv).max(1e-8);
    let tau = p.tau_or_from_dates();
    p.tau = Some(tau + dtau);
    p.eval = None;
    p.expiration = None;
    p
  }
}

/// Poisson-weighted-series Greeks for the Merton (1976) jump-diffusion
/// model.
///
/// `delta`/`gamma`/`rho` are exact closed-form series over the
/// corresponding [`BSMPricer`] Greek (`Σ w_n · greek(σ_n)`): neither the
/// per-term volatility `σ_n` nor the Poisson weights `w_n` depend on spot
/// or rate, so the naive series *is* the true derivative.
/// `vega`/`theta`/`vanna`/`charm`/`volga`/`veta` bump `v`/`tau` on a cloned
/// pricer instead — `σ_n` is itself a function of both (via
/// [`Merton1976Pricer::term_vol`]), so a naive `Σ w_n · greek(σ_n)` would
/// silently drop the chain-rule term and stop being the true derivative of
/// the price. `theta`/`charm`/`veta` use the calendar `-∂/∂τ` convention
/// (matching [`BSMPricer::theta`] / `charm` / `dvega_dtime`, and the
/// `λ ≤ 0` Black-Scholes limit below).
///
/// `theta`/`charm`/`veta`'s `λ > 0` path additionally guards near expiry:
/// at `τ ≤ h_τ` the down-`τ` bump in [`Merton1976Pricer::bumped`] would
/// evaluate the price series at a negative time-to-maturity, producing
/// per-term `NaN`s that [`Merton1976Pricer::greek_series`]'s `NaN`-floor
/// silently zeroes out of the down-leg — turning an undefined derivative
/// into large finite garbage instead of `NaN`. Mirrors
/// [`HestonPricer::theta`](crate::pricing::heston::HestonPricer::theta)'s
/// identical guard.
///
/// All 9 methods price through [`Merton1976Pricer::series_price`], not
/// [`PricerExt::calculate_price`] — so, unlike `calculate_price()` itself,
/// every Greek here stays finite for `m` past `calculate_call_put`'s
/// `usize`-factorial overflow threshold (`m ≈ 21`; see `series_price`'s
/// doc).
impl GreeksExt for Merton1976Pricer {
  fn delta(&self) -> f64 {
    self.greek_series(|bsm| bsm.delta())
  }

  fn gamma(&self) -> f64 {
    self.greek_series(|bsm| bsm.gamma())
  }

  fn rho(&self) -> f64 {
    self.greek_series(|bsm| bsm.rho())
  }

  fn vega(&self) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm(self.tau_or_from_dates()).vega();
    }
    let h = self.h_v();
    (self.bumped(0.0, h, 0.0).series_price() - self.bumped(0.0, -h, 0.0).series_price()) / (2.0 * h)
  }

  fn theta(&self) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm(self.tau_or_from_dates()).theta();
    }
    let tau = self.tau_or_from_dates();
    let h = Self::H_TAU;
    if !(tau.is_finite() && tau > h) {
      return f64::NAN;
    }
    -(self.bumped(0.0, 0.0, h).series_price() - self.bumped(0.0, 0.0, -h).series_price())
      / (2.0 * h)
  }

  fn vanna(&self) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm(self.tau_or_from_dates()).vanna();
    }
    let hs = self.h_s();
    let hv = self.h_v();
    (self.bumped(hs, hv, 0.0).series_price()
      - self.bumped(hs, -hv, 0.0).series_price()
      - self.bumped(-hs, hv, 0.0).series_price()
      + self.bumped(-hs, -hv, 0.0).series_price())
      / (4.0 * hs * hv)
  }

  fn charm(&self) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm(self.tau_or_from_dates()).charm();
    }
    let tau = self.tau_or_from_dates();
    let ht = Self::H_TAU;
    if !(tau.is_finite() && tau > ht) {
      return f64::NAN;
    }
    let hs = self.h_s();
    -(self.bumped(hs, 0.0, ht).series_price()
      - self.bumped(hs, 0.0, -ht).series_price()
      - self.bumped(-hs, 0.0, ht).series_price()
      + self.bumped(-hs, 0.0, -ht).series_price())
      / (4.0 * hs * ht)
  }

  fn volga(&self) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm(self.tau_or_from_dates()).vomma();
    }
    let h = self.h_v();
    let p0 = self.series_price();
    (self.bumped(0.0, h, 0.0).series_price() - 2.0 * p0 + self.bumped(0.0, -h, 0.0).series_price())
      / (h * h)
  }

  fn veta(&self) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm(self.tau_or_from_dates()).dvega_dtime();
    }
    let tau = self.tau_or_from_dates();
    let ht = Self::H_TAU;
    if !(tau.is_finite() && tau > ht) {
      return f64::NAN;
    }
    let hv = self.h_v();
    -(self.bumped(0.0, hv, ht).series_price()
      - self.bumped(0.0, hv, -ht).series_price()
      - self.bumped(0.0, -hv, ht).series_price()
      + self.bumped(0.0, -hv, -ht).series_price())
      / (4.0 * hv * ht)
  }
}

#[cfg(test)]
mod tests;
