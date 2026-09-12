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
#[non_exhaustive]
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
/// strike/maturity grid. The grid itself (`s_max = 3 max(s, k)`, `dt = tau / t_n`)
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
  /// Validating constructor.
  ///
  /// # Panics
  /// Panics for negative or NaN volatility, fewer than one time step, or
  /// fewer than two spot intervals. The implicit schemes require an interior
  /// spot node in addition to the two boundaries. Zero volatility is accepted.
  pub fn new(
    v: f64,
    t_n: usize,
    s_n: usize,
    option_style: OptionStyle,
    method: FiniteDifferenceMethod,
  ) -> Self {
    assert!(
      v >= 0.0,
      "FiniteDifferencePricer::new: v must be a non-negative volatility (got {v})"
    );
    assert!(
      t_n >= 1,
      "FiniteDifferencePricer::new: t_n must be at least 1 (got {t_n})"
    );
    assert!(
      s_n >= 2,
      "FiniteDifferencePricer::new: s_n must be at least 2 (got {s_n})"
    );
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
  /// `american_finite_difference_surface_is_all_nan`.
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
    let s_max = self.s.max(self.k) * 3.0;
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

  fn boundary_condition(&self, s: f64, elapsed: f64) -> f64 {
    let european = match self.option_type {
      OptionType::Call => {
        if s == 0.0 {
          0.0
        } else {
          (s * (-self.q * elapsed).exp() - self.k * (-self.r * elapsed).exp()).max(0.0)
        }
      }
      OptionType::Put => {
        if s == 0.0 {
          self.k * (-self.r * elapsed).exp()
        } else {
          0.0
        }
      }
    };
    match self.model.option_style {
      OptionStyle::European => european,
      OptionStyle::American => european.max(self.payoff(s)),
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
mod tests;
