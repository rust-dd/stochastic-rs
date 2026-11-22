use super::HestonModel;
use super::HestonVanillaCfVegaConfig;
use super::HestonVanillaCfVegaMethod;
use super::VanillaLeg;
use super::VanillaPortfolio;
use super::heston_vanilla_portfolio_cf_initial_variance_vega;
use crate::OptionType;
use crate::pricing::heston::HestonParams;

fn high_vol_of_vol_non_feller_model() -> HestonModel {
  HestonModel {
    s: 490.91,
    params: HestonParams {
      v0: 0.299_494,
      kappa: 4.0,
      theta: 0.072_326,
      sigma: 2.646_148,
      rho: 0.390_534,
    },
    risk_free_rate: 0.05,
    dividend_yield: 0.0,
    tau: 36.0 / 365.0,
  }
}

#[test]
fn non_feller_vertical_cf_vega_is_positive_and_bump_stable() {
  let model = high_vol_of_vol_non_feller_model();
  let payoff = VanillaPortfolio::vertical(OptionType::Put, 455.0, 420.0);
  let estimate = heston_vanilla_portfolio_cf_initial_variance_vega(
    model,
    &payoff,
    HestonVanillaCfVegaConfig::default(),
  )
  .unwrap();

  assert!((estimate.base_price - 7.454_986_5).abs() < 2e-4);
  assert!((estimate.value - 20.849_039).abs() < 2e-4);
  assert!(estimate.bump_stable, "estimate={estimate:?}");
  assert_eq!(estimate.requested_bump, 1e-5);
  assert!((estimate.effective_bump - model.params.v0 * 1e-4).abs() < 1e-15);
  assert_eq!(estimate.comparison_bump, 0.5 * estimate.effective_bump);
}

#[test]
fn cf_vega_is_stable_across_explicit_bumps_without_silent_shrink() {
  let model = high_vol_of_vol_non_feller_model();
  let payoff = VanillaPortfolio::vertical(OptionType::Put, 455.0, 420.0);
  let mut values = Vec::new();
  for bump in [1e-5, 1e-3, 1e-2] {
    let config = HestonVanillaCfVegaConfig {
      initial_variance_bump: bump,
      minimum_relative_initial_variance_bump: 0.0,
      maximum_relative_bump_difference: 1e-3,
    };
    let estimate =
      heston_vanilla_portfolio_cf_initial_variance_vega(model, &payoff, config).unwrap();
    assert_eq!(estimate.effective_bump, bump);
    assert!(estimate.bump_stable, "estimate={estimate:?}");
    values.push(estimate.value);
  }
  let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
  let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  assert!(maximum - minimum < 1e-3, "values={values:?}");
}

#[test]
fn cf_helper_respects_signed_vanilla_portfolios() {
  let model = high_vol_of_vol_non_feller_model();
  let call = VanillaPortfolio::call(455.0);
  let put = VanillaPortfolio::put(455.0);
  let signed = VanillaPortfolio::new(vec![
    VanillaLeg::new(OptionType::Call, 455.0, 2.0),
    VanillaLeg::new(OptionType::Put, 455.0, -1.0),
  ]);
  let config = HestonVanillaCfVegaConfig::default();
  let call_vega = heston_vanilla_portfolio_cf_initial_variance_vega(model, &call, config)
    .unwrap()
    .value;
  let put_vega = heston_vanilla_portfolio_cf_initial_variance_vega(model, &put, config)
    .unwrap()
    .value;
  let signed_vega = heston_vanilla_portfolio_cf_initial_variance_vega(model, &signed, config)
    .unwrap()
    .value;
  assert!((call_vega - put_vega).abs() < 1e-7);
  assert!((signed_vega - (2.0 * call_vega - put_vega)).abs() < 1e-7);
}

#[test]
#[should_panic(expected = "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = 0.0")]
fn cf_helper_panics_on_a_zero_vol_of_vol() {
  let base = high_vol_of_vol_non_feller_model();
  let model = HestonModel {
    params: HestonParams {
      sigma: 0.0,
      ..base.params
    },
    ..base
  };
  let _ = heston_vanilla_portfolio_cf_initial_variance_vega(
    model,
    &VanillaPortfolio::call(455.0),
    HestonVanillaCfVegaConfig::default(),
  );
}

/// Short-dated (`τ = 0.077`), low-variance (`v0 = 0.0109`) put vertical, the hardest inversion:
/// the analytic vega agrees with scipy's `quad` to 6e-9 and with its own bumps to 1e-7 relative.
#[test]
fn short_dated_low_variance_vertical_agrees_with_its_own_bumps() {
  let model = HestonModel {
    s: 720.65,
    params: HestonParams {
      v0: 0.010_896_709_741_918,
      kappa: 8.0,
      theta: 0.054_842_534_327_897_8,
      sigma: 1.258_813_626_480_49,
      rho: -0.479_725_587_697_811,
    },
    risk_free_rate: 0.036_667_089_448_444_35,
    dividend_yield: 0.006_297_374_154_871_42,
    tau: 0.076_712_328_767_123_3,
  };
  let payoff = VanillaPortfolio::vertical(OptionType::Put, 686.0, 649.0);
  let estimate = heston_vanilla_portfolio_cf_initial_variance_vega(
    model,
    &payoff,
    HestonVanillaCfVegaConfig::default(),
  )
  .unwrap();

  assert_eq!(
    estimate.method,
    HestonVanillaCfVegaMethod::AnalyticCharacteristicFunction
  );
  let scipy_quad_reference = 66.200_667_951_845_62;
  assert!((estimate.value - scipy_quad_reference).abs() < 1e-7);
  assert!(estimate.bump_stable, "estimate={estimate:?}");
  assert!(
    estimate.relative_bump_difference < 1e-7,
    "estimate={estimate:?}"
  );
  assert!(
    (estimate.finite_difference_value - estimate.value).abs() < 5e-6,
    "estimate={estimate:?}"
  );
  let diagnostics = [1e-4, 5e-4, 1e-3, 2e-3].map(|bump| {
    heston_vanilla_portfolio_cf_initial_variance_vega(
      model,
      &payoff,
      HestonVanillaCfVegaConfig {
        initial_variance_bump: bump,
        minimum_relative_initial_variance_bump: 0.0,
        maximum_relative_bump_difference: 1e-4,
      },
    )
    .unwrap()
  });
  assert!(diagnostics.iter().all(|diagnostic| diagnostic.bump_stable));
  let errors =
    diagnostics.map(|diagnostic| (diagnostic.finite_difference_value - estimate.value).abs());
  assert!(errors.windows(2).all(|pair| pair[0] < pair[1]));
  // The h² truncation of a central difference at `bump = 1e-4` is `3.5239e-6`; the quadrature
  // adds about `8e-7`.
  assert!(errors[0] < 5e-6, "errors={errors:?}");
  assert!(errors[3] < 2e-3);
  let richardson =
    (4.0 * diagnostics[0].comparison_value - diagnostics[0].finite_difference_value) / 3.0;
  // Richardson removes the `3.5e-6` truncation error but scales the price's `~1e-10` noise by
  // `4/(3·2·5e-5) ≈ 1.3e4`, to `4.0e-7`, which sets the band.
  assert!(
    (richardson - estimate.value).abs() < 1e-6,
    "richardson={richardson:.12} value={:.12}",
    estimate.value
  );
  assert!(diagnostics[3].effective_bump < model.params.v0 / 4.0);
  let resolved = HestonVanillaCfVegaConfig {
    initial_variance_bump: 1e-4,
    minimum_relative_initial_variance_bump: 0.0,
    maximum_relative_bump_difference: 1e-4,
  };
  let body = heston_vanilla_portfolio_cf_initial_variance_vega(
    model,
    &VanillaPortfolio::put(686.0),
    resolved,
  )
  .unwrap();
  let wing = heston_vanilla_portfolio_cf_initial_variance_vega(
    model,
    &VanillaPortfolio::put(649.0),
    resolved,
  )
  .unwrap();
  assert!((estimate.value - (body.value - wing.value)).abs() < 1e-12);
  assert!((body.value - body.finite_difference_value).abs() < 1e-5);
  assert!((wing.value - wing.finite_difference_value).abs() < 1e-5);
}
