use impl_new_derive::ImplNew;
use ndarray::s;
use ndarray::Array1;

use crate::quant::r#trait::PricerExt;
use crate::quant::r#trait::TimeExt;
use crate::quant::OptionStyle;
use crate::quant::OptionType;

#[derive(Default)]
pub enum FiniteDifferenceMethod {
  Explicit,
  Implicit,
  #[default]
  CrankNicolson,
}

#[derive(ImplNew)]
pub struct FiniteDifferencePricer {
  /// Underlying price
  pub s: f64,
  /// Volatility
  pub v: f64,
  /// Strike price
  pub k: f64,
  /// Risk-free rate
  pub r: f64,
  /// Time steps
  pub t_n: usize,
  /// Price steps
  pub s_n: usize,
  /// Time to maturity in years
  pub tau: Option<f64>,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiration: Option<chrono::NaiveDate>,
  /// Otpion style
  pub option_style: OptionStyle,
  /// Option type
  pub option_type: OptionType,
  /// Pricing method
  pub method: FiniteDifferenceMethod,
}

impl PricerExt for FiniteDifferencePricer {
  /// Calculate the option price
  fn calculate_price(&self) -> f64 {
    match self.method {
      FiniteDifferenceMethod::Explicit => self.explicit(),
      FiniteDifferenceMethod::Implicit => self.implicit(),
      FiniteDifferenceMethod::CrankNicolson => self.crank_nicolson(),
    }
  }
}

impl TimeExt for FiniteDifferencePricer {
  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn eval(&self) -> chrono::NaiveDate {
    self.eval.unwrap()
  }

  fn expiration(&self) -> chrono::NaiveDate {
    self.expiration.unwrap()
  }
}

impl FiniteDifferencePricer {
  fn explicit(&self) -> f64 {
    let (dt, ds, s_values, time_steps) = self.calculate_grid();
    let mut option_values = Array1::<f64>::zeros(self.s_n + 1);

    for (i, &s_i) in s_values.iter().enumerate() {
      option_values[i] = self.payoff(s_i);
    }

    for _ in 0..time_steps {
      let mut new_option_values = option_values.clone();

      for i in 1..self.s_n {
        let s_i = s_values[i];

        let delta = (option_values[i + 1] - option_values[i - 1]) / (2.0 * ds);
        let gamma =
          (option_values[i + 1] - 2.0 * option_values[i] + option_values[i - 1]) / (ds.powi(2));

        new_option_values[i] = option_values[i]
          + dt
            * (0.5 * self.v.powi(2) * s_i.powi(2) * gamma + self.r * s_i * delta
              - self.r * option_values[i]);

        if let OptionStyle::American = self.option_style {
          let intrinsic_value = self.payoff(s_i);
          new_option_values[i] = new_option_values[i].max(intrinsic_value);
        }
      }

      new_option_values[0] = self.boundary_condition(s_values[0], 0.0);
      new_option_values[self.s_n] = self.boundary_condition(s_values[self.s_n], 0.0);

      option_values = new_option_values;
    }

    self.interpolate(&s_values, &option_values, self.s)
  }

  fn implicit(&self) -> f64 {
    let (dt, ds, s_values, time_steps) = self.calculate_grid();

    let mut a = Array1::<f64>::zeros(self.s_n - 1);
    let mut b = Array1::<f64>::zeros(self.s_n - 1);
    let mut c = Array1::<f64>::zeros(self.s_n - 1);

    let mut option_values = Array1::<f64>::zeros(self.s_n + 1);
    for (i, &s_i) in s_values.iter().enumerate() {
      option_values[i] = self.payoff(s_i);
    }

    for _ in 0..time_steps {
      for i in 1..self.s_n {
        let s_i = s_values[i];
        let sigma_sq = self.v.powi(2);

        a[i - 1] = -0.5 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) - self.r * s_i / ds);
        b[i - 1] = 1.0 + dt * (sigma_sq * s_i.powi(2) / ds.powi(2) + self.r);
        c[i - 1] = -0.5 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) + self.r * s_i / ds);
      }

      let mut d = option_values.slice(s![1..self.s_n]).to_owned();

      d[0] -= a[0] * self.boundary_condition(0.0, dt);
      d[self.s_n - 2] -= c[self.s_n - 2] * self.boundary_condition(s_values[self.s_n], dt);

      let new_option_values_inner = self.solve_tridiagonal(&a, &b, &c, &d);

      for i in 1..self.s_n {
        option_values[i] = new_option_values_inner[i - 1];

        if let OptionStyle::American = self.option_style {
          let intrinsic_value = self.payoff(s_values[i]);
          option_values[i] = option_values[i].max(intrinsic_value);
        }
      }

      option_values[0] = self.boundary_condition(0.0, dt);
      option_values[self.s_n] = self.boundary_condition(s_values[self.s_n], dt);
    }

    self.interpolate(&s_values, &option_values, self.s)
  }

  fn crank_nicolson(&self) -> f64 {
    let (dt, ds, s_values, time_steps) = self.calculate_grid();

    let mut a = Array1::<f64>::zeros(self.s_n - 1);
    let mut b = Array1::<f64>::zeros(self.s_n - 1);
    let mut c = Array1::<f64>::zeros(self.s_n - 1);

    let mut option_values = Array1::<f64>::zeros(self.s_n + 1);
    for (i, &s_i) in s_values.iter().enumerate() {
      option_values[i] = self.payoff(s_i);
    }

    for _ in 0..time_steps {
      for i in 1..self.s_n {
        let s_i = s_values[i];
        let sigma_sq = self.v.powi(2);

        a[i - 1] = -0.25 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) - self.r * s_i / ds);
        b[i - 1] = 1.0 + 0.5 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) + self.r);
        c[i - 1] = -0.25 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) + self.r * s_i / ds);
      }

      let mut d = Array1::<f64>::zeros(self.s_n - 1);
      for i in 1..self.s_n {
        let s_i = s_values[i];
        let sigma_sq = self.v.powi(2);

        let a_past = 0.25 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) - self.r * s_i / ds);
        let b_past = 1.0 - 0.5 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) + self.r);
        let c_past = 0.25 * dt * (sigma_sq * s_i.powi(2) / ds.powi(2) + self.r * s_i / ds);

        d[i - 1] =
          a_past * option_values[i - 1] + b_past * option_values[i] + c_past * option_values[i + 1];
      }

      d[0] -= a[0] * self.boundary_condition(0.0, dt);
      d[self.s_n - 2] -= c[self.s_n - 2] * self.boundary_condition(s_values[self.s_n], dt);

      let new_option_values_inner = self.solve_tridiagonal(&a, &b, &c, &d);

      for i in 1..self.s_n {
        option_values[i] = new_option_values_inner[i - 1];

        if let OptionStyle::American = self.option_style {
          let intrinsic_value = self.payoff(s_values[i]);
          option_values[i] = option_values[i].max(intrinsic_value);
        }
      }

      option_values[0] = self.boundary_condition(0.0, dt);
      option_values[self.s_n] = self.boundary_condition(s_values[self.s_n], dt);
    }

    self.interpolate(&s_values, &option_values, self.s)
  }

  fn calculate_grid(&self) -> (f64, f64, Array1<f64>, usize) {
    let tau = self.tau.unwrap_or(1.0);
    let dt = tau / self.t_n as f64;
    let s_max = self.s * 3.0;
    let ds = s_max / self.s_n as f64;
    let s_values = Array1::linspace(0.0, s_max, self.s_n + 1);
    let time_steps = self.t_n;
    (dt, ds, s_values, time_steps)
  }

  fn payoff(&self, s: f64) -> f64 {
    match self.option_type {
      OptionType::Call => (s - self.k).max(0.0),
      OptionType::Put => (self.k - s).max(0.0),
    }
  }

  fn boundary_condition(&self, s: f64, tau: f64) -> f64 {
    match self.option_type {
      OptionType::Call => {
        if s == 0.0 {
          0.0
        } else {
          s - self.k * (-self.r * (self.tau.unwrap_or(1.0) - tau)).exp()
        }
      }
      OptionType::Put => {
        if s == 0.0 {
          self.k * (-self.r * (self.tau.unwrap_or(1.0) - tau)).exp()
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

  fn solve_tridiagonal(
    &self,
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
}

#[cfg(test)]
mod tests {
  use super::FiniteDifferenceMethod;
  use super::FiniteDifferencePricer;
  use crate::quant::r#trait::PricerExt;
  use crate::quant::OptionStyle;
  use crate::quant::OptionType;
  use crate::stochastic::K;
  use crate::stochastic::S0;

  fn atm_pricer(style: OptionStyle, r#type: OptionType, method: FiniteDifferenceMethod) -> f64 {
    let pricer = FiniteDifferencePricer::new(
      S0,
      0.1,
      K,
      0.05,
      10000,
      250,
      Some(1.0),
      None,
      None,
      style,
      r#type,
      method,
    );

    pricer.calculate_price()
  }

  #[test]
  fn eu_explicit_call() {
    let call = atm_pricer(
      OptionStyle::European,
      OptionType::Call,
      FiniteDifferenceMethod::Explicit,
    );
    println!("Call: {}", call);
  }

  #[test]
  fn eu_implicit_call() {
    let call = atm_pricer(
      OptionStyle::European,
      OptionType::Call,
      FiniteDifferenceMethod::Implicit,
    );
    println!("Call: {}", call);
  }

  #[test]
  fn eu_crank_nicolson_call() {
    let call = atm_pricer(
      OptionStyle::European,
      OptionType::Call,
      FiniteDifferenceMethod::CrankNicolson,
    );
    println!("Call: {}", call);
  }

  #[test]
  fn am_explicit_call() {
    let call = atm_pricer(
      OptionStyle::American,
      OptionType::Call,
      FiniteDifferenceMethod::Explicit,
    );
    println!("Call: {}", call);
  }

  #[test]
  fn am_implicit_call() {
    let call = atm_pricer(
      OptionStyle::American,
      OptionType::Call,
      FiniteDifferenceMethod::Implicit,
    );
    println!("Call: {}", call);
  }

  #[test]
  fn am_crank_nicolson_call() {
    let call = atm_pricer(
      OptionStyle::American,
      OptionType::Call,
      FiniteDifferenceMethod::CrankNicolson,
    );
    println!("Call: {}", call);
  }

  #[test]
  fn eu_explicit_put() {
    let put = atm_pricer(
      OptionStyle::European,
      OptionType::Put,
      FiniteDifferenceMethod::Explicit,
    );
    println!("Put: {}", put);
  }

  #[test]
  fn eu_implicit_put() {
    let put = atm_pricer(
      OptionStyle::European,
      OptionType::Put,
      FiniteDifferenceMethod::Implicit,
    );
    println!("Put: {}", put);
  }

  #[test]
  fn eu_crank_nicolson_put() {
    let put = atm_pricer(
      OptionStyle::European,
      OptionType::Put,
      FiniteDifferenceMethod::CrankNicolson,
    );
    println!("Put: {}", put);
  }

  #[test]
  fn am_explicit_put() {
    let put = atm_pricer(
      OptionStyle::American,
      OptionType::Put,
      FiniteDifferenceMethod::Explicit,
    );
    println!("Put: {}", put);
  }

  #[test]
  fn am_implicit_put() {
    let put = atm_pricer(
      OptionStyle::American,
      OptionType::Put,
      FiniteDifferenceMethod::Implicit,
    );
    println!("Put: {}", put);
  }

  #[test]
  fn am_crank_nicolson_put() {
    let put = atm_pricer(
      OptionStyle::American,
      OptionType::Put,
      FiniteDifferenceMethod::CrankNicolson,
    );
    println!("Put: {}", put);
  }
}
