//! Bivariate-copula + portfolio Value-at-Risk demo.
//!
//! Models the joint distribution of two equity log-returns with a Clayton
//! copula (lower-tail dependence — captures correlated draw-downs) and
//! computes the **portfolio 1-day 99 % VaR** for an equal-weight book of
//! the two assets.
//!
//! Pipeline:
//!   1. Build the copula with the requested Kendall-τ.
//!   2. Sample (u, v) pairs in [0,1]².
//!   3. Push each margin through a `N(μ, σ)` inverse-CDF to recover
//!      realistic equity-return marginals.
//!   4. Aggregate into a portfolio P&L sample.
//!   5. Apply both Gaussian-parametric and historical VaR estimators and
//!      compare against the i.i.d. Gaussian baseline that ignores the
//!      tail dependence.
//!
//! Run with:
//!   cargo run --release --example copula_var_demo

use ndarray::Array1;
use stochastic_rs::copulas::bivariate::clayton::Clayton;
use stochastic_rs::quant::risk::PnlOrLoss;
use stochastic_rs::quant::risk::VarMethod;
use stochastic_rs::quant::risk::value_at_risk;
use stochastic_rs::traits::BivariateExt;

/// Inverse standard-normal CDF via Beasley-Springer-Moro — good enough for
/// the demo (no statrs dep needed). Errors at ±3σ are < 1e-8.
fn norm_ppf(p: f64) -> f64 {
  // Beasley-Springer-Moro algorithm constants.
  const A: [f64; 4] = [
    2.50662823884,
    -18.61500062529,
    41.39119773534,
    -25.44106049637,
  ];
  const B: [f64; 4] = [
    -8.47351093090,
    23.08336743743,
    -21.06224101826,
    3.13082909833,
  ];
  const C: [f64; 9] = [
    0.3374754822726147,
    0.9761690190917186,
    0.1607979714918209,
    0.0276438810333863,
    0.0038405729373609,
    0.0003951896511919,
    0.0000321767881768,
    0.0000002888167364,
    0.0000003960315187,
  ];
  let y = p - 0.5;
  if y.abs() < 0.42 {
    let r = y * y;
    return y * (((A[3] * r + A[2]) * r + A[1]) * r + A[0])
      / ((((B[3] * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0);
  }
  let r = if y < 0.0 { p } else { 1.0 - p };
  let r = (-r.ln()).ln();
  // Polynomial Σ C[k] · r^k → Horner's from the highest-order coefficient
  // down: ((C[8]·r + C[7])·r + C[6])·r + … + C[0].
  let mut x = C[8];
  for &c in C[..8].iter().rev() {
    x = x * r + c;
  }
  if y < 0.0 { -x } else { x }
}

fn main() {
  println!("=== stochastic-rs Clayton-copula portfolio VaR demo ===\n");

  let n_samples = 50_000usize;
  let tau = 0.4f64; // Kendall-τ → moderate lower-tail dependence
  let mu = [0.0005, 0.0003]; // daily drift
  let sigma = [0.015, 0.020]; // daily vol
  let weights = [0.5f64, 0.5];
  let confidence = 0.99;

  // 1) Clayton copula with the requested τ → θ.
  let mut clayton = Clayton::new();
  clayton.set_tau(tau);
  clayton._compute_theta();
  let theta = clayton.theta().expect("theta computed from tau");
  println!("Clayton(τ = {tau}) → θ = {theta:.4}");

  // 2) Sample (u, v) ∈ [0,1]² with a fixed seed.
  let uv = clayton
    .sample_with_seed(n_samples, 42)
    .expect("Clayton sample");

  // 3) Push through marginal inverse CDFs and convert to per-asset PnL.
  let mut pnl: Array1<f64> = Array1::zeros(n_samples);
  for i in 0..n_samples {
    let u = uv[[i, 0]].clamp(1e-9, 1.0 - 1e-9);
    let v = uv[[i, 1]].clamp(1e-9, 1.0 - 1e-9);
    let r1 = mu[0] + sigma[0] * norm_ppf(u);
    let r2 = mu[1] + sigma[1] * norm_ppf(v);
    pnl[i] = weights[0] * r1 + weights[1] * r2;
  }

  // 4) VaR estimates.
  let gauss_var = value_at_risk(pnl.view(), confidence, PnlOrLoss::Pnl, VarMethod::Gaussian);
  let hist_var = value_at_risk(
    pnl.view(),
    confidence,
    PnlOrLoss::Pnl,
    VarMethod::Historical,
  );

  // Baseline: i.i.d. Gaussian portfolio (no copula structure).
  let port_mu = weights[0] * mu[0] + weights[1] * mu[1];
  let port_var = weights[0].powi(2) * sigma[0].powi(2) + weights[1].powi(2) * sigma[1].powi(2);
  let port_sigma = port_var.sqrt();
  let iid_var = -port_mu + port_sigma * norm_ppf(confidence);

  println!("\nPortfolio (1-day, 99% VaR, equal-weight):");
  println!("  Gaussian VaR on Clayton-sampled P&L : {gauss_var:.5}");
  println!("  Historical VaR on Clayton-sampled P&L: {hist_var:.5}");
  println!("  Baseline i.i.d. Gaussian VaR         : {iid_var:.5}");
  println!(
    "\nObservation: tail-dependent copulas typically inflate the historical \
     VaR relative to the i.i.d. Gaussian baseline because joint draw-downs \
     happen more often than the bivariate normal predicts."
  );
}
