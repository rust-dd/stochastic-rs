//! # Finite Difference
//!
//! $$
//! \partial_t V+\tfrac12\sigma^2S^2\partial_{SS}V+(r-q)S\partial_SV-rV=0
//! $$
//!
use ndarray::Array1;
use ndarray::s;

use crate::OptionStyle;
use crate::OptionType;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FiniteDifferenceMethod {
  Explicit,
  Implicit,
  #[default]
  CrankNicolson,
}

/// Black-Scholes PDE solver on a uniform spot/time grid.
///
/// The struct holds **model and method state only** — the volatility, the
/// grid sizes, the exercise style and the time-stepping scheme. Spot,
/// strike, rate, dividend yield, maturity and the option direction are the
/// pricing *query* and travel as arguments to
/// [`ModelPricer::price_call`], so one instance prices a whole
/// strike/maturity grid. The grid itself (`s_max = 3s`, `dt = tau / t_n`)
/// is rebuilt inside every call from the query, so nothing derived from a
/// spot or a maturity is cached across queries.
///
/// ```
/// use stochastic_rs_quant::pricing::finite_difference::FiniteDifferencePricer;
/// use stochastic_rs_quant::traits::ModelPricer;
/// use stochastic_rs_quant::OptionStyle;
/// use stochastic_rs_quant::pricing::finite_difference::FiniteDifferenceMethod;
///
/// let model = FiniteDifferencePricer::new(
///     0.25, 500, 100, OptionStyle::American, FiniteDifferenceMethod::CrankNicolson);
/// let american = model.price_put(100.0, 105.0, 0.05, 0.02, 0.75);
/// let european = FiniteDifferencePricer::new(
///     0.25, 500, 100, OptionStyle::European, FiniteDifferenceMethod::CrankNicolson)
///     .price_put(100.0, 105.0, 0.05, 0.02, 0.75);
/// assert!(american >= european);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FiniteDifferencePricer {
  /// Volatility
  pub v: f64,
  /// Time steps
  pub t_n: usize,
  /// Price steps
  pub s_n: usize,
  /// Option style
  pub option_style: OptionStyle,
  /// Pricing method
  pub method: FiniteDifferenceMethod,
}

impl FiniteDifferencePricer {
  pub const fn new(
    v: f64,
    t_n: usize,
    s_n: usize,
    option_style: OptionStyle,
    method: FiniteDifferenceMethod,
  ) -> Self {
    Self {
      v,
      t_n,
      s_n,
      option_style,
      method,
    }
  }

  /// Solve the PDE for one option direction at one query point.
  pub fn price(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    let solve = FdSolve {
      model: self,
      s,
      k,
      r,
      q,
      tau,
      option_type,
    };
    match self.method {
      FiniteDifferenceMethod::Explicit => solve.explicit(),
      FiniteDifferenceMethod::Implicit => solve.implicit(),
      FiniteDifferenceMethod::CrankNicolson => solve.crank_nicolson(),
    }
  }
}

impl ModelPricer for FiniteDifferencePricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price(s, k, r, q, tau, OptionType::Call)
  }

  /// Overrides the trait's vanilla-parity default: this solver prices the
  /// put by solving the same PDE against the put payoff, which is what the
  /// pre-query `calculate_price()` did. Parity would be wrong outright at
  /// [`OptionStyle::American`] (the put's early-exercise premium), and even
  /// at `European` it would return the *call's* discretisation error
  /// reflected rather than the put's own. See
  /// `fd_price_put_overrides_vanilla_parity`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price(s, k, r, q, tau, OptionType::Put)
  }
}

/// A European vanilla call **only at** [`OptionStyle::European`]; the same
/// solver at [`OptionStyle::American`] returns an American price, which the
/// Black inversion has no volatility to offer. The exercise style is a field
/// rather than a type parameter, so this is the one in-tree implementor whose
/// answer depends on the instance.
impl VanillaEuropeanCall for FiniteDifferencePricer {
  /// $Se^{(r-q)\tau}$ at [`OptionStyle::European`], and [`f64::NAN`] at
  /// [`OptionStyle::American`] — case 2 of [the failure
  /// convention](ModelPricer#how-pricing-fails).
  ///
  /// Without the `NaN` an American surface is not merely approximate but
  /// convincing: at `q = 0.06` every point on a 5x2 grid inverts to a finite
  /// vol within 0.008 of the model's own `v`, so nothing in the output marks
  /// it as an American price pushed through a European formula. See
  /// `fd_american_surface_is_all_nan`.
  fn vanilla_call_forward(&self, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    match self.option_style {
      OptionStyle::European => s * ((r - q) * tau).exp(),
      OptionStyle::American => f64::NAN,
    }
  }
}

/// One PDE solve: the model plus the query it is being evaluated at.
///
/// Private and short-lived — it exists so the three time-stepping schemes
/// and their grid/payoff/boundary helpers can read `self.k` and `self.r`
/// the way they did before the query moved out of the pricer, instead of
/// threading six extra arguments through each of them. It is not an API
/// shape: nothing outside this module can name it, and it is constructed
/// fresh per call.
struct FdSolve<'a> {
  model: &'a FiniteDifferencePricer,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
  option_type: OptionType,
}

impl FdSolve<'_> {
  fn explicit(&self) -> f64 {
    let (dt, ds, s_values, time_steps) = self.calculate_grid();
    let s_n = self.model.s_n;
    let mut option_values = Array1::<f64>::zeros(s_n + 1);

    for (i, &s_i) in s_values.iter().enumerate() {
      option_values[i] = self.payoff(s_i);
    }

    for step in 0..time_steps {
      let mut new_option_values = option_values.clone();
      let elapsed = (step + 1) as f64 * dt;

      for i in 1..s_n {
        let s_i = s_values[i];

        let delta = (option_values[i + 1] - option_values[i - 1]) / (2.0 * ds);
        let gamma =
          (option_values[i + 1] - 2.0 * option_values[i] + option_values[i - 1]) / (ds.powi(2));

        new_option_values[i] = option_values[i]
          + dt
            * (0.5 * self.model.v.powi(2) * s_i.powi(2) * gamma + self.drift() * s_i * delta
              - self.r * option_values[i]);

        if let OptionStyle::American = self.model.option_style {
          let intrinsic_value = self.payoff(s_i);
          new_option_values[i] = new_option_values[i].max(intrinsic_value);
        }
      }

      new_option_values[0] = self.boundary_condition(s_values[0], elapsed);
      new_option_values[s_n] = self.boundary_condition(s_values[s_n], elapsed);

      option_values = new_option_values;
    }

    self.interpolate(&s_values, &option_values, self.s)
  }

  fn implicit(&self) -> f64 {
    let (dt, ds, s_values, time_steps) = self.calculate_grid();
    let s_n = self.model.s_n;

    let mut a = Array1::<f64>::zeros(s_n - 1);
    let mut b = Array1::<f64>::zeros(s_n - 1);
    let mut c = Array1::<f64>::zeros(s_n - 1);

    let mut option_values = Array1::<f64>::zeros(s_n + 1);
    for (i, &s_i) in s_values.iter().enumerate() {
      option_values[i] = self.payoff(s_i);
    }

    for step in 0..time_steps {
      let elapsed = (step + 1) as f64 * dt;

      for i in 1..s_n {
        let s_i = s_values[i];
        let sigma_sq = self.model.v.powi(2);

        a[i - 1] = -0.5 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) - self.drift() * s_i / ds);
        b[i - 1] = 1.0 + dt * (sigma_sq * s_i.powi(2) / ds.powi(2) + self.r);
        c[i - 1] = -0.5 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) + self.drift() * s_i / ds);
      }

      let mut d = option_values.slice(s![1..s_n]).to_owned();

      d[0] -= a[0] * self.boundary_condition(0.0, elapsed);
      d[s_n - 2] -= c[s_n - 2] * self.boundary_condition(s_values[s_n], elapsed);

      let new_option_values_inner = solve_tridiagonal(&a, &b, &c, &d);

      for i in 1..s_n {
        option_values[i] = new_option_values_inner[i - 1];

        if let OptionStyle::American = self.model.option_style {
          let intrinsic_value = self.payoff(s_values[i]);
          option_values[i] = option_values[i].max(intrinsic_value);
        }
      }

      option_values[0] = self.boundary_condition(0.0, elapsed);
      option_values[s_n] = self.boundary_condition(s_values[s_n], elapsed);
    }

    self.interpolate(&s_values, &option_values, self.s)
  }

  fn crank_nicolson(&self) -> f64 {
    let (dt, ds, s_values, time_steps) = self.calculate_grid();
    let s_n = self.model.s_n;

    let mut a = Array1::<f64>::zeros(s_n - 1);
    let mut b = Array1::<f64>::zeros(s_n - 1);
    let mut c = Array1::<f64>::zeros(s_n - 1);

    let mut option_values = Array1::<f64>::zeros(s_n + 1);
    for (i, &s_i) in s_values.iter().enumerate() {
      option_values[i] = self.payoff(s_i);
    }

    for step in 0..time_steps {
      let elapsed = (step + 1) as f64 * dt;

      for i in 1..s_n {
        let s_i = s_values[i];
        let sigma_sq = self.model.v.powi(2);

        a[i - 1] = -0.25 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) - self.drift() * s_i / ds);
        b[i - 1] = 1.0 + 0.5 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) + self.r);
        c[i - 1] = -0.25 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) + self.drift() * s_i / ds);
      }

      let mut d = Array1::<f64>::zeros(s_n - 1);
      for i in 1..s_n {
        let s_i = s_values[i];
        let sigma_sq = self.model.v.powi(2);

        let a_past = 0.25 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) - self.drift() * s_i / ds);
        let b_past = 1.0 - 0.5 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) + self.r);
        let c_past = 0.25 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) + self.drift() * s_i / ds);

        d[i - 1] =
          a_past * option_values[i - 1] + b_past * option_values[i] + c_past * option_values[i + 1];
      }

      d[0] -= a[0] * self.boundary_condition(0.0, elapsed);
      d[s_n - 2] -= c[s_n - 2] * self.boundary_condition(s_values[s_n], elapsed);

      let new_option_values_inner = solve_tridiagonal(&a, &b, &c, &d);

      for i in 1..s_n {
        option_values[i] = new_option_values_inner[i - 1];

        if let OptionStyle::American = self.model.option_style {
          let intrinsic_value = self.payoff(s_values[i]);
          option_values[i] = option_values[i].max(intrinsic_value);
        }
      }

      option_values[0] = self.boundary_condition(0.0, elapsed);
      option_values[s_n] = self.boundary_condition(s_values[s_n], elapsed);
    }

    self.interpolate(&s_values, &option_values, self.s)
  }

  /// Risk-neutral drift of the underlying, `r - q`. Before the query moved
  /// out of the pricer there was no dividend-yield input at all and this
  /// term was plain `r`; `q = 0` reproduces that exactly.
  fn drift(&self) -> f64 {
    self.r - self.q
  }

  fn calculate_grid(&self) -> (f64, f64, Array1<f64>, usize) {
    let dt = self.tau / self.model.t_n as f64;
    let s_max = self.s * 3.0;
    let ds = s_max / self.model.s_n as f64;
    let s_values = Array1::linspace(0.0, s_max, self.model.s_n + 1);
    let time_steps = self.model.t_n;
    (dt, ds, s_values, time_steps)
  }

  fn payoff(&self, s: f64) -> f64 {
    match self.option_type {
      OptionType::Call => (s - self.k).max(0.0),
      OptionType::Put => (self.k - s).max(0.0),
    }
  }

  fn boundary_condition(&self, s: f64, tau: f64) -> f64 {
    let remaining = self.tau - tau;
    match self.option_type {
      OptionType::Call => {
        if s == 0.0 {
          0.0
        } else {
          s * (-self.q * remaining).exp() - self.k * (-self.r * remaining).exp()
        }
      }
      OptionType::Put => {
        if s == 0.0 {
          self.k * (-self.r * remaining).exp()
        } else {
          0.0
        }
      }
    }
  }

  fn interpolate(&self, s_values: &Array1<f64>, option_values: &Array1<f64>, s: f64) -> f64 {
    for i in 0..s_values.len() - 1 {
      if s_values[i] <= s && s <= s_values[i + 1] {
        let weight = (s - s_values[i]) / (s_values[i + 1] - s_values[i]);
        return option_values[i] * (1.0 - weight) + option_values[i + 1] * weight;
      }
    }
    0.0
  }
}

fn solve_tridiagonal(
  a: &Array1<f64>,
  b: &Array1<f64>,
  c: &Array1<f64>,
  d: &Array1<f64>,
) -> Array1<f64> {
  let n = d.len();
  let mut c_star = Array1::<f64>::zeros(n);
  let mut d_star = Array1::<f64>::zeros(n);

  c_star[0] = c[0] / b[0];
  d_star[0] = d[0] / b[0];

  for i in 1..n {
    let m = b[i] - a[i] * c_star[i - 1];
    c_star[i] = c[i] / m;
    d_star[i] = (d[i] - a[i] * d_star[i - 1]) / m;
  }

  let mut x = Array1::<f64>::zeros(n);
  x[n - 1] = d_star[n - 1];
  for i in (0..n - 1).rev() {
    x[i] = d_star[i] - c_star[i] * x[i + 1];
  }

  x
}

#[cfg(test)]
mod tests {
  use stochastic_rs_stochastic::K;
  use stochastic_rs_stochastic::S0;

  use super::*;

  fn atm_pricer(style: OptionStyle, r#type: OptionType, method: FiniteDifferenceMethod) -> f64 {
    FiniteDifferencePricer::new(0.1, 10000, 250, style, method).price(S0, K, 0.05, 0.0, 1.0, r#type)
  }

  #[test]
  fn eu_explicit_call() {
    let call = atm_pricer(
      OptionStyle::European,
      OptionType::Call,
      FiniteDifferenceMethod::Explicit,
    );
    assert!(call.is_finite() && call > 0.0);
  }

  #[test]
  fn eu_implicit_call() {
    let call = atm_pricer(
      OptionStyle::European,
      OptionType::Call,
      FiniteDifferenceMethod::Implicit,
    );
    assert!(call.is_finite() && call > 0.0);
  }

  #[test]
  fn eu_crank_nicolson_call() {
    let call = atm_pricer(
      OptionStyle::European,
      OptionType::Call,
      FiniteDifferenceMethod::CrankNicolson,
    );
    assert!(call.is_finite() && call > 0.0);
  }

  #[test]
  fn am_explicit_call() {
    let call = atm_pricer(
      OptionStyle::American,
      OptionType::Call,
      FiniteDifferenceMethod::Explicit,
    );
    assert!(call.is_finite() && call > 0.0);
  }

  #[test]
  fn am_implicit_call() {
    let call = atm_pricer(
      OptionStyle::American,
      OptionType::Call,
      FiniteDifferenceMethod::Implicit,
    );
    assert!(call.is_finite() && call > 0.0);
  }

  #[test]
  fn am_crank_nicolson_call() {
    let call = atm_pricer(
      OptionStyle::American,
      OptionType::Call,
      FiniteDifferenceMethod::CrankNicolson,
    );
    assert!(call.is_finite() && call > 0.0);
  }

  #[test]
  fn eu_explicit_put() {
    let put = atm_pricer(
      OptionStyle::European,
      OptionType::Put,
      FiniteDifferenceMethod::Explicit,
    );
    assert!(put.is_finite() && put > 0.0);
  }

  #[test]
  fn eu_implicit_put() {
    let put = atm_pricer(
      OptionStyle::European,
      OptionType::Put,
      FiniteDifferenceMethod::Implicit,
    );
    assert!(put.is_finite() && put > 0.0);
  }

  #[test]
  fn eu_crank_nicolson_put() {
    let put = atm_pricer(
      OptionStyle::European,
      OptionType::Put,
      FiniteDifferenceMethod::CrankNicolson,
    );
    assert!(put.is_finite() && put > 0.0);
  }

  #[test]
  fn am_explicit_put() {
    let put = atm_pricer(
      OptionStyle::American,
      OptionType::Put,
      FiniteDifferenceMethod::Explicit,
    );
    assert!(put.is_finite() && put > 0.0);
  }

  #[test]
  fn am_implicit_put() {
    let put = atm_pricer(
      OptionStyle::American,
      OptionType::Put,
      FiniteDifferenceMethod::Implicit,
    );
    assert!(put.is_finite() && put > 0.0);
  }

  #[test]
  fn am_crank_nicolson_put() {
    let put = atm_pricer(
      OptionStyle::American,
      OptionType::Put,
      FiniteDifferenceMethod::CrankNicolson,
    );
    assert!(put.is_finite() && put > 0.0);
  }

  const S: f64 = 100.0;
  const KK: f64 = 105.0;
  const R: f64 = 0.05;
  const TAU: f64 = 0.75;
  const V: f64 = 0.25;

  /// Cross-arch tolerance: 500 time steps of `exp`/tridiagonal arithmetic
  /// accumulate a last bit that differs between aarch64-darwin and CI's
  /// ubuntu x86_64.
  const TOL: f64 = 1e-12;

  /// Captured from `PricerExt::calculate_price()` **before** the
  /// `ModelPricer` reshape, at `t_n = 500, s_n = 100` and **`q = 0`** —
  /// the only dividend yield the pre-query pricer could express, since it
  /// had no `q` field. Every one of the twelve `(method, style, type)`
  /// combinations is pinned, so the `q` term added to the PDE is proven
  /// inert at `q = 0` rather than merely believed to be.
  #[test]
  fn fd_model_pricer_matches_pre_refactor_goldens_at_zero_q() {
    let cases: &[(FiniteDifferenceMethod, OptionStyle, OptionType, f64)] = &[
      (
        FiniteDifferenceMethod::Explicit,
        OptionStyle::European,
        OptionType::Call,
        8.11461770850247,
      ),
      (
        FiniteDifferenceMethod::Explicit,
        OptionStyle::European,
        OptionType::Put,
        9.249887489204166,
      ),
      (
        FiniteDifferenceMethod::Explicit,
        OptionStyle::American,
        OptionType::Call,
        8.11461770850247,
      ),
      (
        FiniteDifferenceMethod::Explicit,
        OptionStyle::American,
        OptionType::Put,
        9.821824693576902,
      ),
      (
        FiniteDifferenceMethod::Implicit,
        OptionStyle::European,
        OptionType::Call,
        8.110495234182721,
      ),
      (
        FiniteDifferenceMethod::Implicit,
        OptionStyle::European,
        OptionType::Put,
        9.246048837934325,
      ),
      (
        FiniteDifferenceMethod::Implicit,
        OptionStyle::American,
        OptionType::Call,
        8.110495234182721,
      ),
      (
        FiniteDifferenceMethod::Implicit,
        OptionStyle::American,
        OptionType::Put,
        9.813744792158452,
      ),
      (
        FiniteDifferenceMethod::CrankNicolson,
        OptionStyle::European,
        OptionType::Call,
        8.112556939207138,
      ),
      (
        FiniteDifferenceMethod::CrankNicolson,
        OptionStyle::European,
        OptionType::Put,
        9.24796864891483,
      ),
      (
        FiniteDifferenceMethod::CrankNicolson,
        OptionStyle::American,
        OptionType::Call,
        8.112556939207138,
      ),
      (
        FiniteDifferenceMethod::CrankNicolson,
        OptionStyle::American,
        OptionType::Put,
        9.817753373035876,
      ),
    ];
    for &(method, style, ot, expected) in cases {
      let got =
        FiniteDifferencePricer::new(V, 500, 100, style, method).price(S, KK, R, 0.0, TAU, ot);
      assert!(
        (got - expected).abs() < TOL,
        "{method:?}/{style:?}/{ot:?}: got {got}, want {expected}"
      );
    }
  }

  /// The dividend-yield term this task added to the PDE actually moves the
  /// price, in the direction and roughly the magnitude Black-Scholes says
  /// it should. Without this, `q` could be threaded through and silently
  /// ignored — the `pricing/slv.rs` failure mode.
  #[test]
  fn fd_dividend_yield_drives_the_price() {
    let model = FiniteDifferencePricer::new(
      V,
      500,
      100,
      OptionStyle::European,
      FiniteDifferenceMethod::CrankNicolson,
    );
    let no_div = model.price_call(S, KK, R, 0.0, TAU);
    let with_div = model.price_call(S, KK, R, 0.08, TAU);
    assert!(
      with_div < no_div - 1.0,
      "a large dividend yield must cut the call materially: {with_div} vs {no_div}"
    );

    use crate::pricing::bsm::BSMCoc;
    use crate::pricing::bsm::BSMPricer;
    let bs = BSMPricer::new(V, BSMCoc::Merton1973).price_call(S, KK, R, 0.08, TAU);
    assert!(
      (with_div - bs).abs() < 0.1,
      "European FD with q must track Black-Scholes with the same q: {with_div} vs {bs}"
    );
  }

  /// American exercise binds for a call once the dividend yield exceeds the
  /// rate — the case that is unreachable without a `q` input, and the
  /// reason the pre-query pricer's American and European calls were always
  /// equal.
  #[test]
  fn fd_american_call_beats_european_under_dividends() {
    let eu = FiniteDifferencePricer::new(
      V,
      500,
      100,
      OptionStyle::European,
      FiniteDifferenceMethod::CrankNicolson,
    )
    .price_call(S, KK, 0.03, 0.10, TAU);
    let am = FiniteDifferencePricer::new(
      V,
      500,
      100,
      OptionStyle::American,
      FiniteDifferenceMethod::CrankNicolson,
    )
    .price_call(S, KK, 0.03, 0.10, TAU);
    assert!(am > eu, "american {am} must exceed european {eu}");
  }

  /// The trait's European put-call parity is the wrong answer for an
  /// American solve.
  #[test]
  fn fd_price_put_overrides_vanilla_parity() {
    let model = FiniteDifferencePricer::new(
      V,
      500,
      100,
      OptionStyle::American,
      FiniteDifferenceMethod::CrankNicolson,
    );
    let call = model.price_call(S, KK, R, 0.0, TAU);
    let put = model.price_put(S, KK, R, 0.0, TAU);
    let vanilla = call - S + KK * (-R * TAU).exp();
    assert!(
      put > vanilla + 1e-3,
      "American put must exceed the European-parity value: {put} vs {vanilla}"
    );
  }

  /// The capability the reshape exists for: one model, a whole grid.
  #[test]
  fn fd_one_model_prices_a_grid() {
    let model = FiniteDifferencePricer::new(
      V,
      200,
      80,
      OptionStyle::European,
      FiniteDifferenceMethod::CrankNicolson,
    );
    for &tau in &[0.25, 0.5, 1.0] {
      let mut prev = f64::INFINITY;
      for &k in &[90.0, 100.0, 110.0] {
        let c = model.price_call(S, k, R, 0.02, tau);
        assert!(c.is_finite() && c < prev, "call must fall in strike");
        prev = c;
      }
    }
  }
}
