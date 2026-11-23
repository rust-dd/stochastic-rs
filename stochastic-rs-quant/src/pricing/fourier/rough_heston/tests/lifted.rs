use ndarray::array;
use num_complex::Complex64;
use stochastic_rs_distributions::special::gamma;

use super::super::LogChf;
use super::super::RoughHestonParams;
use super::super::chf;
use super::super::cumulants;
use super::super::lifted::LiftedCf;
use super::ratios;
use crate::pricing::fourier::FourierModelExt;
use crate::pricing::fourier::HestonFourier;
use crate::pricing::heston::HestonParams;

/// Abi Jaber's (3.7).
const AJ: RoughHestonParams = RoughHestonParams {
  v0: 0.02,
  kappa: 0.3,
  theta: 0.02,
  nu: 0.3,
  rho: -0.7,
  hurst: 0.1,
};

fn twenty() -> LiftedCf {
  LiftedCf::new(AJ, 20, 2.5, 300)
}

/// "for n = 20, r₂₀ = 2.5 and H = 0.1, xⁿ_n = 6417.74" (App. C.1), to his two decimals, and the
/// bound `xᵢ ≥ r^{i−1−n/2}` of §3.2.
/// Reference: Abi Jaber (2019), Quant. Finance 19(12), 1995–2013, DOI 10.1080/14697688.2019.1615113
#[test]
fn the_published_largest_mean_reversion_is_reproduced() {
  let cf = twenty();
  assert!(
    (cf.reversions[19] - 6417.74).abs() < 5e-3,
    "{}",
    cf.reversions[19]
  );
  for (i, x) in cf.reversions.iter().enumerate() {
    assert!(*x >= 2.5_f64.powf(i as f64 - 10.0), "x[{i}] = {x}");
  }
  assert!(cf.reversions[0] < 1e-3 && cf.weights.iter().all(|c| *c > 0.0));
}

/// The 20 factors hold the fractional kernel within 4.8 % on `[0.01, 1]`; the bound is twice that.
#[test]
fn the_factors_approximate_the_fractional_kernel() {
  let cf = twenty();
  let alpha = AJ.alpha();
  for t in [0.01_f64, 0.1, 1.0] {
    let lifted = cf
      .weights
      .iter()
      .zip(&cf.reversions)
      .map(|(c, x)| c * (-x * t).exp());
    let ratio = lifted.sum::<f64>() / (t.powf(alpha - 1.0) / gamma(alpha));
    assert!((ratio - 1.0).abs() < 0.1, "t = {t}: {ratio}");
  }
}

/// One factor of weight 1 and no mean reversion is Heston; (C.2) is first order.
#[test]
fn one_unit_factor_is_heston_at_first_order() {
  let p = HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.05,
    sigma: 0.3,
    rho: -0.7,
  };
  let rough = RoughHestonParams {
    v0: p.v0,
    kappa: p.kappa,
    theta: p.theta,
    nu: p.sigma,
    rho: p.rho,
    hurst: 0.25,
  };
  let heston = HestonFourier::new(p);
  for xi in [
    Complex64::new(0.5, -0.5),
    Complex64::new(3.0, -0.5),
    Complex64::new(3.0, 0.0),
  ] {
    let want = heston.chf(1.0, xi).ln();
    let error = |steps: usize| {
      let cf = LiftedCf::from_kernel(rough, array![1.0], array![0.0], steps);
      (cf.log_chf(1.0, Complex64::i() * xi) - want).norm()
    };
    let errors = [300, 600, 1200].map(error);
    assert!(errors[2] < 2e-3 * want.norm(), "{xi}: {errors:?}");
    for ratio in ratios(&errors) {
      assert!((1.8..2.2).contains(&ratio), "{xi}: {errors:?}");
    }
  }
}

/// At `Δ·μ·Σcᵢ/(2 + xᵢΔ) = 1` the linearised step has the eigenvalue −1; a solve keeps half of it.
#[test]
fn the_step_count_keeps_the_explicit_term_inside_its_stability_bound() {
  let cf = twenty();
  let dt = 1e-3;
  let mu = cf.load(dt).recip();
  let mode = cf.reversions.mapv(|x| dt * mu / (2.0 + x * dt));
  let total = cf.weights.dot(&mode);
  assert!((total - 1.0).abs() < 1e-14, "{total}");
  let mut psi = mode.mapv(Complex64::from).to_vec();
  cf.advance(&mut psi, dt, Complex64::from(-mu * total));
  for (after, before) in psi.iter().zip(&mode) {
    assert!(
      (after + before).norm() < 1e-13 * before.abs(),
      "{after} vs {before}"
    );
  }
  for mu in [1.0, 1e3, 1e4] {
    let steps = cf.steps_for(1.0, mu).expect("within the cap");
    assert!(
      steps >= 300 && mu * cf.load(1.0 / steps as f64) <= 0.5,
      "μ = {mu}: {steps}"
    );
  }
  assert_eq!(cf.steps_for(1.0, 1e-3), Some(300));
  assert_eq!(cf.steps_for(1.0, f64::NAN), None);
  assert_eq!(cf.steps_for(1.0, 1e12), None);
  assert!(chf(&cf, 0.25, Complex64::new(1e4, -0.5)).is_finite());
  assert!(chf(&cf, 0.25, Complex64::new(1e13, -0.5)).is_nan());
}

#[test]
fn the_lifted_chf_follows_the_domain_rule_and_its_cumulants_are_its_own() {
  let cf = twenty();
  let xi = Complex64::new(3.0, -0.5);
  assert_eq!(chf(&cf, 1.0, Complex64::new(0.0, -1.0)), Complex64::ONE);
  assert_eq!(chf(&cf, 0.0, xi), Complex64::ONE);
  assert!(chf(&cf, -1.0, xi).is_nan() && chf(&cf, f64::NAN, xi).is_nan());
  assert!((chf(&cf, 1.0, -xi.conj()) - chf(&cf, 1.0, xi).conj()).norm() < 1e-13);
  let got = cumulants(&cf, 1.0);
  let f = |u: f64| cf.log_chf(1.0, Complex64::new(0.0, u));
  let h = 2e-3;
  let (f2p, fp, f0, fm, f2m) = (f(2.0 * h), f(h), f(0.0), f(-h), f(-2.0 * h));
  let c1 = (-f2p + 8.0 * fp - 8.0 * fm + f2m).im / (12.0 * h);
  let c2 = -(-f2p + 16.0 * fp - 30.0 * f0 + 16.0 * fm - f2m).re / (12.0 * h * h);
  assert!((got.c1 - c1).abs() < 1e-8, "{} vs {c1}", got.c1);
  assert!((got.c2 - c2).abs() < 1e-7, "{} vs {c2}", got.c2);
  // At `V₀ = θ` the mean variance is flat, so `c₁ = −½V₀t`; (C.2), first order, leaves a relative
  // 6.6e-5 at 300 steps, taken three times.
  assert!((got.c1 / (-0.5 * AJ.v0) - 1.0).abs() < 2e-4, "{}", got.c1);
}
