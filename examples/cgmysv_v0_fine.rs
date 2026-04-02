use stochastic_rs::quant::pricing::cgmysv::{CgmysvModel, CgmysvParams};
use stochastic_rs::quant::pricing::fourier::LewisPricer;

fn main() {
  let tau = 28.0 / 365.0;
  let s0 = 2488.11;
  let k = 2500.0_f64;
  let r = 0.01213;
  let q = 0.01884;

  for v0_x10000 in 100..=115 {
    let v0 = v0_x10000 as f64 / 10000.0;
    let params = CgmysvParams {
      alpha: 0.5184, lambda_plus: 25.4592, lambda_minus: 4.6040,
      kappa: 1.0029, eta: 0.0711, zeta: 0.3443, rho: -2.0283, v0,
    };
    let model = CgmysvModel { params, r, q };
    let call = LewisPricer::price_call(&model, s0, k, r, q, tau);
    let marker = if (call - 19.659).abs() < 0.15 { " <== MATCH" } else { "" };
    println!("v0={v0:.4}  call={call:.4}{marker}");
  }
}
