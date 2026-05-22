use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use super::pricer::BSMPricer;
use crate::OptionType;
use crate::traits::PricerExt;

impl BSMPricer {
  /// Calculate the delta
  pub fn delta(&self) -> f64 {
    let (d1, _) = self.d1_d2();
    let tau = self.tau_required();
    let exp_bt = ((self.b() - self.r) * tau).exp();

    if self.option_type == OptionType::Call {
      exp_bt * norm_cdf(d1)
    } else {
      exp_bt * (norm_cdf(d1) - 1.0)
    }
  }

  /// Calculate the gamma
  pub fn gamma(&self) -> f64 {
    let T = self.tau_required();
    let (d1, _) = self.d1_d2();

    ((self.b() - self.r) * T).exp() * norm_pdf(d1) / (self.s * self.v * self.tau_required().sqrt())
  }

  /// Calculate the gamma percent
  pub fn gamma_percent(&self) -> f64 {
    self.gamma() / self.s * 100.0
  }

  /// Calculate the theta
  pub fn theta(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    let exp_bt = ((self.b() - self.r) * self.tau_required()).exp();
    let exp_rt = (-self.r * self.tau_required()).exp();
    let pdf_d1 = norm_pdf(d1);

    let first_term = -self.s * exp_bt * pdf_d1 * self.v / (2.0 * self.tau_required().sqrt());

    if self.option_type == OptionType::Call {
      let second_term = -(self.b() - self.r) * self.s * exp_bt * norm_cdf(d1);
      let third_term = -self.r * self.k * exp_rt * norm_cdf(d2);
      first_term + second_term + third_term
    } else {
      let second_term = (self.b() - self.r) * self.s * exp_bt * norm_cdf(-d1);
      let third_term = -self.r * self.k * exp_rt * norm_cdf(-d2);
      first_term + second_term + third_term
    }
  }

  /// Calculate the vega
  pub fn vega(&self) -> f64 {
    let (d1, _) = self.d1_d2();

    self.s
      * ((self.b() - self.r) * self.tau_required()).exp()
      * norm_pdf(d1)
      * self.tau_required().sqrt()
  }

  /// Calculate the rho
  pub fn rho(&self) -> f64 {
    let (_, d2) = self.d1_d2();

    let exp_rt = (-self.r * self.tau_required()).exp();

    if self.option_type == OptionType::Call {
      self.k * self.tau_required() * exp_rt * norm_cdf(d2)
    } else {
      -self.k * self.tau_required() * exp_rt * norm_cdf(-d2)
    }
  }

  /// Calculate the vomma
  pub fn vomma(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    self.vega() * d1 * d2 / self.v
  }

  /// Calculate the charm
  pub fn charm(&self) -> f64 {
    let v = self.v;
    let r = self.r;
    let b = self.b();
    let tau = self.tau_required();
    let (d1, d2) = self.d1_d2();

    let exp_bt = ((b - r) * tau).exp();
    let pdf_d1 = norm_pdf(d1);
    let sqrt_T = tau.sqrt();

    match self.option_type {
      OptionType::Call => {
        exp_bt * (pdf_d1 * ((b / (v * sqrt_T)) - (d2 / (2.0 * tau))) + (b - r) * norm_cdf(d1))
      }
      OptionType::Put => {
        exp_bt * (pdf_d1 * ((b / (v * sqrt_T)) - (d2 / (2.0 * tau))) - (b - r) * norm_cdf(-d1))
      }
    }
  }

  /// Calculate the vanna
  pub fn vanna(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    -((self.b() - self.r) * self.tau_required()).exp() * norm_pdf(d1) * d2 / self.v
  }

  /// Calculate the zomma
  pub fn zomma(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    self.gamma() * (d1 * d2 - 1.0) / self.v
  }

  /// Calculate the zomma percent
  pub fn zomma_percent(&self) -> f64 {
    self.zomma() * self.s / 100.0
  }

  /// Calculate the speed
  pub fn speed(&self) -> f64 {
    let (d1, _) = self.d1_d2();

    -self.gamma() * (1.0 + d1 / (self.v * self.tau_required().sqrt())) / self.s
  }

  /// Calculate the color
  pub fn color(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    self.gamma()
      * (self.r - self.b()
        + self.b() * d1 / (self.v * self.tau_required().sqrt())
        + (1.0 - d1 * d2) / (2.0 * self.tau_required()))
  }

  /// Calculate the ultima
  pub fn ultima(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    -self.vomma() / self.v * (d1 * d2 - (d1 / d2) + (d2 / d1) - 1.0)
  }

  /// Calculate the DvegaDtime
  pub fn dvega_dtime(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    self.vega()
      * (self.r - self.b() + self.b() * d1 / (self.v * self.tau_required().sqrt())
        - (d1 * d2 + 1.0) / (2.0 * self.tau_required()))
  }

  /// Calculating Lambda (elasticity)
  pub fn lambda(&mut self) -> (f64, f64) {
    let (call, put) = self.calculate_call_put();
    (self.delta() * self.s / call, self.delta() * self.s / put)
  }

  /// Calculate the phi
  pub fn phi(&self) -> f64 {
    let (d1, _) = self.d1_d2();

    let exp_bt = ((self.b() - self.r) * self.tau_required()).exp();

    if self.option_type == OptionType::Call {
      -self.tau_required() * self.s * exp_bt * norm_cdf(d1)
    } else {
      self.tau_required() * self.s * exp_bt * norm_cdf(-d1)
    }
  }

  /// Calculate the zeta
  pub fn zeta(&self) -> f64 {
    let (_, d2) = self.d1_d2();

    if self.option_type == OptionType::Call {
      norm_cdf(d2)
    } else {
      -norm_cdf(-d2)
    }
  }

  /// Calculate the strike delta
  pub fn strike_delta(&self) -> f64 {
    let (_, d2) = self.d1_d2();

    let exp_rt = (-self.r * self.tau_required()).exp();

    if self.option_type == OptionType::Call {
      -exp_rt * norm_cdf(d2)
    } else {
      exp_rt * norm_cdf(-d2)
    }
  }

  /// Calculate the strike gamma
  pub fn strike_gamma(&self) -> f64 {
    let (_, d2) = self.d1_d2();

    norm_pdf(d2) * (-self.r * self.tau_required()).exp()
      / (self.k * self.v * self.tau_required().sqrt())
  }
}
