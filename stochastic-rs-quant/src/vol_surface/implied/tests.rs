use ndarray::array;

use super::*;

/// `from_provider` against a mock chain of present-value mids, the OTM put at K = 90: the
/// recovered vols must equal the σ that priced them.
#[test]
fn from_provider_recovers_surface_from_mock_chain() {
  use stochastic_rs_distributions::special::norm_cdf;

  use crate::market::provider::ChainQuote;
  use crate::market::provider::MockProvider;
  use crate::market::provider::OptionChain;

  let s = 100.0_f64;
  let r = 0.04_f64;
  let sigma = 0.22_f64;
  let strikes = [90.0_f64, 100.0, 110.0];
  let taus = [0.5_f64, 1.0];

  let mut quotes = Vec::new();
  for &t in &taus {
    let f = s * (r * t).exp();
    for &k in &strikes {
      let d1 = ((f / k).ln() + 0.5 * sigma * sigma * t) / (sigma * t.sqrt());
      let d2 = d1 - sigma * t.sqrt();
      let option_type = if k > 90.0 {
        OptionType::Call
      } else {
        OptionType::Put
      };
      let undiscounted = if option_type == OptionType::Call {
        f * norm_cdf(d1) - k * norm_cdf(d2)
      } else {
        k * norm_cdf(-d2) - f * norm_cdf(-d1)
      };
      let price = (-r * t).exp() * undiscounted;
      quotes.push(ChainQuote {
        strike: k,
        tau: t,
        last: price,
        bid: price,
        ask: price,
        implied_vol: sigma,
        option_type,
      });
    }
  }
  let mut mp = MockProvider::new();
  mp.insert_option_chain(OptionChain {
    symbol: "ACME".to_string(),
    s,
    quotes,
  });

  let surface = ImpliedVolSurface::from_provider(&mp, "ACME", r, 0.0).unwrap();
  for (j, tau) in taus.iter().enumerate() {
    for (i, k) in strikes.iter().enumerate() {
      let iv = surface.ivs[[j, i]];
      assert!(
        (iv - sigma).abs() < 1e-10,
        "recovered iv={iv} vs σ={sigma} at T={tau}, K={k}"
      );
    }
  }
}

/// `from_provider` propagates a provider error for a missing symbol.
#[test]
fn from_provider_errors_on_missing_symbol() {
  use crate::market::provider::MockProvider;

  let mp = MockProvider::new();
  let res = ImpliedVolSurface::from_provider(&mp, "NOPE", 0.05, 0.0);
  assert!(res.is_err());
}

#[test]
fn from_prices_round_trip() {
  use stochastic_rs_distributions::special::norm_cdf;

  let s = 100.0;
  let r = 0.05;
  let sigma = 0.20;

  let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
  let maturities = vec![0.25, 0.50, 1.0];
  let forwards: Vec<f64> = maturities.iter().map(|&t| s * f64::exp(r * t)).collect();

  let mut prices = Array2::<f64>::zeros((maturities.len(), strikes.len()));
  for (j, &t) in maturities.iter().enumerate() {
    let f = forwards[j];
    for (i, &k) in strikes.iter().enumerate() {
      let d1 = ((f / k).ln() + 0.5 * sigma * sigma * t) / (sigma * t.sqrt());
      let d2 = d1 - sigma * t.sqrt();
      // Undiscounted Black price
      prices[[j, i]] = f * norm_cdf(d1) - k * norm_cdf(d2);
    }
  }

  let surface = ImpliedVolSurface::from_prices(
    strikes.clone(),
    maturities.clone(),
    forwards.clone(),
    prices.view(),
    OptionType::Call,
  );

  for (j, tau) in maturities.iter().enumerate() {
    for (i, k) in strikes.iter().enumerate() {
      let iv = surface.ivs[[j, i]];
      assert!(
        (iv - sigma).abs() < 1e-6,
        "iv={iv} vs sigma={sigma} at T={tau}, K={k}"
      );
    }
  }
}

#[test]
fn smile_slice_filters_nans() {
  let surface = ImpliedVolSurface {
    strikes: vec![90.0, 100.0, 110.0],
    maturities: vec![0.5],
    forwards: vec![100.0],
    ivs: array![[0.22, f64::NAN, 0.20]],
    total_variance: array![[0.0242, f64::NAN, 0.02]],
    log_moneyness: array![[(90.0_f64 / 100.0).ln(), 0.0, (110.0_f64 / 100.0).ln()]],
  };

  let slice = surface.smile_slice(0);
  assert_eq!(slice.implied_vols.len(), 2);
  assert!((slice.implied_vols[0] - 0.22).abs() < 1e-12);
  assert!((slice.implied_vols[1] - 0.20).abs() < 1e-12);
}

/// Round trips against mpmath at 50 digits, the deep tail included, and the expiry and bound cases.
#[test]
fn black_implied_vol_round_trips_the_reference_table() {
  let cases = [
    (7.965567455405797, 100.0, 100.0, 1.0, 0.2),
    (2.6361352697261307, 100.0, 120.0, 0.75, 0.25),
    (27.123141927196656, 100.0, 80.0, 2.0, 0.3),
    (0.005794111542994946, 100.0, 150.0, 0.05, 0.6),
    (0.7978712629263207, 100.0, 100.0, 0.01, 0.2),
    (3.4529165077418788e-28, 100.0, 300.0, 0.25, 0.2),
  ];
  for (price, f, k, t, sigma) in cases {
    let iv = black_implied_vol(price, f, k, t, OptionType::Call);
    assert!(
      (iv - sigma).abs() < 1e-11,
      "k = {k}, t = {t}: {iv} vs {sigma}"
    );
  }
  assert_eq!(
    black_implied_vol(10.0, 100.0, 90.0, 0.0, OptionType::Call),
    0.0
  );
  assert!(black_implied_vol(10.5, 100.0, 90.0, 0.0, OptionType::Call).is_nan());
  assert_eq!(
    black_implied_vol(60.0, 100.0, 40.0, 0.1, OptionType::Call),
    0.0
  );
  assert_eq!(
    black_implied_vol(100.0, 100.0, 40.0, 0.1, OptionType::Call),
    f64::INFINITY
  );
  assert!(black_implied_vol(101.0, 100.0, 40.0, 0.1, OptionType::Call).is_nan());
}
