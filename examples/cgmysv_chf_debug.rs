use num_complex::Complex64;
use stochastic_rs::quant::pricing::cgmysv::{CgmysvModel, CgmysvParams};
use stochastic_rs::quant::pricing::fourier::FourierModelExt;

fn main() {
  let params = CgmysvParams {
    alpha: 0.5184,
    lambda_plus: 25.4592,
    lambda_minus: 4.6040,
    kappa: 1.0029,
    eta: 0.0711,
    zeta: 0.3443,
    rho: -2.0283,
    v0: 0.006381,
  };
  let model = CgmysvModel { params: params.clone(), r: 0.01213, q: 0.01884 };
  let tau = 28.0 / 365.0;

  println!("=== ChF diagnostics (t = {tau:.6}) ===");
  println!();

  // 1. phi(0) should be 1
  let phi0 = model.chf(tau, Complex64::new(0.0, 0.0));
  println!("phi(0)         = {:.10} + {:.10}i  (should be 1+0i)", phi0.re, phi0.im);

  // 2. phi(-i) should be exp((r-q)*tau) for risk-neutral
  let phi_neg_i = model.chf(tau, Complex64::new(0.0, -1.0));
  let expected = ((0.01213 - 0.01884) * tau).exp();
  println!("phi(-i)        = {:.10} + {:.10}i", phi_neg_i.re, phi_neg_i.im);
  println!("exp((r-q)T)    = {:.10}  (should match Re(phi(-i)))", expected);

  // 3. |phi(u)| for real u — must be <= 1
  println!();
  println!("u        |phi(u)|       Re(phi)        Im(phi)");
  for &u in &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0] {
    let phi = model.chf(tau, Complex64::new(u, 0.0));
    println!("{u:<8} {:<14.8} {:<14.8} {:<14.8}", phi.norm(), phi.re, phi.im);
  }

  // 4. omega
  let omega = params.omega(tau);
  println!();
  println!("omega(T)       = {omega:.10}");
  println!("E[exp(L_T)]    = {:.10}", omega.exp());

  // 5. psi_stdCGMY at key points
  println!();
  println!("psi_stdCGMY(0) = {:?}", params.psi_std_cgmy(Complex64::new(0.0, 0.0)));
  println!("psi_stdCGMY(1) = {:?}", params.psi_std_cgmy(Complex64::new(1.0, 0.0)));
  println!("psi_stdCGMY(-i)= {:?}", params.psi_std_cgmy(Complex64::new(0.0, -1.0)));

  // 6. CIR transform at trivial points
  println!();
  let phi_cir_00 = params.cir_transform(tau, Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), params.v0);
  println!("CIR(0, 0, v0)  = {:?}  (should be 1)", phi_cir_00);

  // 7. Gil-Pelaez integrand samples
  println!();
  println!("Gil-Pelaez P2 integrand samples (should decay smoothly):");
  let i_unit = Complex64::i();
  let ln_ks = (2500.0_f64 / 2488.11).ln();
  for &u in &[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] {
    let xi = Complex64::new(u, 0.0);
    let phi = model.chf(tau, xi);
    let kernel = (-i_unit * u * ln_ks).exp() / (i_unit * u);
    let val = (kernel * phi).re;
    println!("  u={u:<6} integrand={val:<14.8}  |phi|={:.8}", phi.norm());
  }
}
