use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_quant::calibration::rbergomi::RBergomiParams;
use stochastic_rs_quant::calibration::rbergomi::RBergomiXi0;
use stochastic_rs_quant::calibration::rbergomi::bid_ask_tolerance;
use stochastic_rs_quant::pricing::malliavin_thalmaier::MultiHestonPaths;
use stochastic_rs_stats::fou::FouFit;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::heston::Euler;
use stochastic_rs_stochastic::volatility::heston::Heston;

/// The Hurst exponent transfers, `η = |σ|`, `ξ₀` a flat `|μ|` and `ρ = −0.7`; each held off its
/// bound: `H` in `[1e-3, 0.499]`, `η ≥ 1e-8`, `ξ₀ ≥ 1e-4`.
#[test]
fn seed_from_fou_maps_the_fractional_ou_fit() {
  let seed =
    |hurst, sigma, mu| RBergomiParams::seed_from_fou(FouFit::from((hurst, sigma, mu, 2.0)));
  let start = seed(0.12, -0.8, -0.04);
  assert_eq!((start.hurst, start.rho, start.eta), (0.12, -0.7, 0.8));
  assert!(matches!(start.xi0, RBergomiXi0::Constant(xi) if xi == 0.04));
  assert_eq!(seed(0.7, 0.3, 0.02).hurst, 0.499);
  assert_eq!(seed(0.0, 0.3, 0.02).hurst, 1e-3);
  assert_eq!(seed(0.1, 0.0, 0.02).eta, 1e-8);
  assert!(matches!(seed(0.1, 0.3, 0.0).xi0, RBergomiXi0::Constant(xi) if xi == 1e-4));
}

/// The mean bid-ask width `Σ|askⱼ − bidⱼ| / M`; NaN over no quote, where no mean exists.
#[test]
fn bid_ask_tolerance_is_the_mean_spread() {
  let (bid, ask) = ([1.0, 2.0, 3.5], [1.1, 2.3, 3.4]);
  let mean = ((1.1_f64 - 1.0).abs() + (2.3_f64 - 2.0).abs() + (3.4_f64 - 3.5).abs()) / 3.0;
  assert!((bid_ask_tolerance(&bid, &ask) - mean).abs() < 1e-15);
  assert!(bid_ask_tolerance(&[], &[]).is_nan());
}

#[test]
#[should_panic(
  expected = "ask must satisfy `ask.len() == bid.len()`, got ask.len() = 1, bid.len() = 2"
)]
fn bid_ask_tolerance_needs_one_ask_per_bid() {
  bid_ask_tolerance(&[1.0, 2.0], &[1.1]);
}

/// Row `i` is Heston `i`'s own seeded path; Euler's arithmetic update keeps the paths out of the
/// conditional Malliavin weights.
#[test]
fn from_hestons_stacks_each_hestons_path() {
  let heston = |seed: u64, s0: f64| {
    Heston::<f64, Euler, _>::new(
      s0,
      0.04,
      1.5,
      0.04,
      0.3,
      -0.6,
      0.02,
      17,
      Some(1.0),
      HestonPow::Sqrt,
      VarianceBoundary::FullTruncation,
      Deterministic::new(seed),
    )
  };
  let hestons = [heston(3, 100.0), heston(4, 80.0)];
  let paths = MultiHestonPaths::from_hestons(&hestons.clone());
  assert_eq!(paths.n_assets, 2);
  for (i, h) in hestons.iter().enumerate() {
    let [s, v] = h.sample();
    assert_eq!(paths.prices.row(i), s.view(), "prices of asset {i}");
    assert_eq!(paths.vols.row(i), v.view(), "variances of asset {i}");
  }
  assert!(!paths.supports_conditional_malliavin_weights());
}
