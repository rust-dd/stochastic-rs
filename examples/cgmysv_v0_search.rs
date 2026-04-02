use stochastic_rs::quant::pricing::cgmysv::{CgmysvModel, CgmysvParams};
use stochastic_rs::quant::pricing::fourier::{CarrMadanPricer, LewisPricer};

fn main() {
  let cm = CarrMadanPricer::default();
  let tau = 28.0 / 365.0;
  let s0 = 2488.11;
  let k = 2500.0_f64;
  let r = 0.01213;
  let q = 0.01884;

  println!("v0           Lewis_call   (target: 19.6590)");
  for v0_x1000 in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60] {
    let v0 = v0_x1000 as f64 / 1000.0;
    let params = CgmysvParams {
      alpha: 0.5184, lambda_plus: 25.4592, lambda_minus: 4.6040,
      kappa: 1.0029, eta: 0.0711, zeta: 0.3443, rho: -2.0283, v0,
    };
    let model = CgmysvModel { params, r, q };
    let lw = LewisPricer::price_call(&model, s0, k, r, q, tau);
    let marker = if (lw - 19.659).abs() < 1.0 { " <== CLOSE" } else { "" };
    println!("{v0:<12.6}  {lw:>10.4}{marker}");
  }
}
