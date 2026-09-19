use stochastic_rs_quant::OptionStyle;
use stochastic_rs_quant::pricing::bsm::BSMCoc;
use stochastic_rs_quant::pricing::bsm::BSMPricer;
use stochastic_rs_quant::pricing::finite_difference::FiniteDifferenceMethod;
use stochastic_rs_quant::pricing::finite_difference::FiniteDifferencePricer;
use stochastic_rs_quant::pricing::variance_swap::VarianceSwapPricer;
use stochastic_rs_quant::traits::ModelPricer;

#[test]
fn finite_difference_discounts_deep_in_the_money_puts() {
  let expected = BSMPricer::new(0.2, BSMCoc::Merton1973).price_put(1.0, 100.0, 0.1, 0.0, 1.0);
  for method in [
    FiniteDifferenceMethod::Implicit,
    FiniteDifferenceMethod::CrankNicolson,
  ] {
    let pricer = FiniteDifferencePricer::new(0.2, 2000, 300, OptionStyle::European, method);
    let actual = pricer.price_put(1.0, 100.0, 0.1, 0.0, 1.0);
    assert!(
      (actual - expected).abs() < 0.01,
      "{method:?}: {actual} vs {expected}"
    );
  }
}

#[test]
fn discrete_variance_swap_recovers_constant_variance_log_return_moment() {
  for (r, q) in [(0.05, 0.0), (0.05, 0.02), (0.0, 0.03)] {
    let pricer = VarianceSwapPricer {
      s: 100.0,
      r,
      q,
      tau: 1.0,
    };
    let variance = 0.04;
    let expected = variance + (r - q - 0.5 * variance).powi(2) / 12.0;
    let actual = pricer.fair_strike_heston_discrete(variance, 2.0, variance, 0.0, 0.0, 12);
    assert!(
      (actual - expected).abs() < 1e-14,
      "r={r}, q={q}: {actual} vs {expected}"
    );
  }
}

/// Independent 60-digit quadrature of the CIR second moment in the
/// Bernard–Cui first-order coefficient (arXiv:1305.7092, Proposition 6.1).
#[test]
fn discrete_variance_swap_matches_cir_moments_and_zero_reversion_limit() {
  let pricer = VarianceSwapPricer {
    s: 100.0,
    r: 0.05,
    q: 0.02,
    tau: 2.0,
  };
  for (kappa, rho, expected) in [
    (1.5, -0.7, 0.055_886_655_204_077_44),
    (1.5, 0.0, 0.055_840_124_468_915_88),
    (1.5, 0.7, 0.055_793_593_733_754_32),
    (1e-11, -0.7, 0.090_092_857_142_356_39),
    (0.0, -0.7, 0.090_092_857_142_857_14),
  ] {
    let actual = pricer.fair_strike_heston_discrete(0.09, kappa, 0.04, 0.3, rho, 252);
    assert!(
      (actual - expected).abs() < 1e-14,
      "kappa={kappa}, rho={rho}: {actual} vs {expected}"
    );
  }
}
