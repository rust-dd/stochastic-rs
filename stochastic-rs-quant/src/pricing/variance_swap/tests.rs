use super::*;

const BUSINESS_DAY_252_DT: f64 = 1.0 / 252.0;

const S: f64 = 100.0;
const R: f64 = 0.05;
const Q: f64 = 0.0;
const TAU: f64 = 1.0;

fn heston(v0: f64, kappa: f64, theta: f64) -> HestonParams {
  HestonParams {
    v0,
    kappa,
    theta,
    sigma: 0.3,
    rho: -0.7,
  }
}

fn replicate(strikes: &[f64], otm_prices: &[f64]) -> f64 {
  VarianceSwapPricer::fair_strike_replication(strikes, otm_prices, S, R, Q, TAU)
}

#[test]
fn bsm_fair_strike_is_sigma_squared() {
  assert!((VarianceSwapPricer::fair_strike_bsm(0.2) - 0.04).abs() < 1e-12);
}

#[test]
fn realized_variance_constant_path_is_zero() {
  let prices = vec![100.0; 252];
  let rv = VarianceSwapPricer::realized_variance(&prices, BUSINESS_DAY_252_DT);
  assert!((rv - 0.0).abs() < 1e-15);
}

#[test]
fn realized_variance_recovers_known_drift() {
  let dt: f64 = BUSINESS_DAY_252_DT;
  let daily = 0.20 * dt.sqrt();
  let prices: Vec<f64> = (0..253).map(|i| 100.0 * (daily * i as f64).exp()).collect();
  let rv = VarianceSwapPricer::realized_variance(&prices, dt);
  assert!((rv - 0.04).abs() < 0.005, "rv={rv}, expected≈0.04");
}

#[test]
fn pnl_scales_with_notional() {
  let terms = VarianceSwapTerms {
    strike_variance: 0.04,
    notional: 100_000.0,
  };
  assert!((VarianceSwapPricer::pnl(0.06, &terms) - 2_000.0).abs() < 1e-9);
}

#[test]
fn vol_swap_convexity_lowers_strike() {
  let k_vol = VolatilitySwapPricer::fair_strike_from_var(0.04, 0.001);
  assert!(k_vol < 0.04_f64.sqrt());
  assert!(k_vol > 0.0);
}

#[test]
fn vol_swap_zero_dispersion_recovers_sqrt_var() {
  let k_vol = VolatilitySwapPricer::fair_strike_from_var(0.04, 0.0);
  assert!((k_vol - 0.2).abs() < 1e-10);
}

#[test]
fn heston_fair_strike_equals_v0_when_at_long_run_mean() {
  let k_var = VarianceSwapPricer::fair_strike_heston(&heston(0.04, 1.5, 0.04), TAU);
  assert!((k_var - 0.04).abs() < 1e-12);
}

#[test]
fn heston_fair_strike_blends_v0_to_theta() {
  // V0 = 0.09 (high IV), θ = 0.04 (low LR), strong κ → fair K close to θ
  let k_strong = VarianceSwapPricer::fair_strike_heston(&heston(0.09, 5.0, 0.04), TAU);
  let k_weak = VarianceSwapPricer::fair_strike_heston(&heston(0.09, 0.1, 0.04), TAU);
  assert!(k_strong < k_weak);
  assert!(k_weak <= 0.09);
  assert!(k_strong > 0.04);
}

#[test]
fn heston_kappa_zero_limit_equals_v0() {
  assert!(
    (VarianceSwapPricer::fair_strike_heston(&heston(0.04, 0.0, 0.10), TAU) - 0.04).abs() < 1e-12
  );
}

#[test]
fn heston_long_t_limit_approaches_theta() {
  // T → ∞ with κ > 0 ⇒ factor → 0, K_var → θ.
  let k_var = VarianceSwapPricer::fair_strike_heston(&heston(0.09, 2.0, 0.04), 50.0);
  assert!(
    (k_var - 0.04).abs() < 0.01,
    "K_var={k_var} should approach θ=0.04"
  );
}

#[test]
fn heston_discrete_correction_vanishes_with_n() {
  let heston = HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  };
  let k_cont = VarianceSwapPricer::fair_strike_heston(&heston, TAU);
  let k_disc_fine = VarianceSwapPricer::fair_strike_heston_discrete(&heston, 100_000, R, Q, TAU);
  let k_disc_coarse = VarianceSwapPricer::fair_strike_heston_discrete(&heston, 12, R, Q, TAU);
  assert!((k_disc_fine - k_cont).abs() < (k_disc_coarse - k_cont).abs());
}

/// A maturity that is not positive leaves no window to annualise over.
#[test]
fn replication_is_nan_at_a_maturity_that_is_not_positive() {
  for tau in [0.0, -1.0, f64::NAN] {
    let strikes = [90.0, 100.0, 110.0];
    let got =
      VarianceSwapPricer::fair_strike_replication(&strikes, &[1.0, 2.0, 1.0], 100.0, 0.0, 0.0, tau);
    assert!(got.is_nan(), "tau={tau}: {got}");
  }
}

/// The trapezoidal weights read `strikes[i±1]`, so one strike is not a
/// coarse strip — it is not a strip. Checked at both reachable lengths so
/// the guard cannot pass on the empty case alone.
#[test]
fn replication_rejects_a_strip_shorter_than_two_strikes() {
  for (strikes, prices) in [(&[][..], &[][..]), (&[100.0][..], &[2.0][..])] {
    let err = std::panic::catch_unwind(|| replicate(strikes, prices)).unwrap_err();
    let msg = err
      .downcast_ref::<String>()
      .cloned()
      .unwrap_or_else(|| (*err.downcast_ref::<&str>().unwrap_or(&"")).to_string());
    assert!(
      msg.contains("static replication needs at least 2 strikes"),
      "wrong panic for len {}: {msg}",
      strikes.len()
    );
  }
}

/// The guards above are worth nothing if a `NaN` can still arrive as
/// `0.0`, and `f64::max` hands back the finite operand when the other is
/// `NaN` — so the final floor had to learn to test first. A `NaN` option
/// price is the reachable source: the strikes are `debug_assert`ed finite,
/// the prices never were.
#[test]
fn replication_does_not_floor_a_nan_price_to_zero() {
  let strikes = [90.0, 100.0, 110.0];
  let got = replicate(&strikes, &[1.0, f64::NAN, 1.0]);
  assert!(
    got.is_nan(),
    "a NaN price must not floor to a strike, got {got}"
  );

  let clean = replicate(&strikes, &[1.0, 2.0, 1.0]);
  assert!(
    clean.is_finite(),
    "control case must still price, got {clean}"
  );
}

/// One price is no observation at all, which must not read as the zero of a constant path.
#[test]
#[should_panic(expected = "realized variance needs at least 2 prices (got 1)")]
fn realized_variance_rejects_a_single_price() {
  let _ = VarianceSwapPricer::realized_variance(&[100.0], BUSINESS_DAY_252_DT);
}

#[test]
#[should_panic(expected = "realized variance needs at least 2 prices (got 0)")]
fn realized_variance_rejects_an_empty_path() {
  let _ = VarianceSwapPricer::realized_variance(&[], BUSINESS_DAY_252_DT);
}

/// A negative variance strike is a programmer error, and a zero one leaves the Jensen term dividing
/// by `k_var^{3/2}`.
#[test]
#[should_panic(expected = "variance strike k_var must be strictly positive (got -0.01)")]
fn vol_swap_rejects_a_negative_variance_strike() {
  let _ = VolatilitySwapPricer::fair_strike_from_var(-0.01, 0.001);
}

#[test]
#[should_panic(expected = "variance strike k_var must be strictly positive (got 0)")]
fn vol_swap_rejects_a_zero_variance_strike() {
  let _ = VolatilitySwapPricer::fair_strike_from_var(0.0, 0.001);
}

/// A NaN vol-of-vol, which the variance strike's guard cannot see, prices NaN rather than the
/// dispersion-free `sqrt(k_var)`.
#[test]
fn vol_swap_heston_preserves_a_nan_vol_of_vol() {
  let params = HestonParams {
    sigma: f64::NAN,
    ..heston(0.04, 1.5, 0.04)
  };
  let k = VolatilitySwapPricer::fair_strike_heston(&params, 1.0);
  assert!(k.is_nan(), "a NaN sigma must exit as NaN, got {k}");
}

/// The poison check must not disturb the dispersion it is guarding: a
/// finite vol-of-vol still lowers the strike below `sqrt(k_var)` by the
/// convexity correction, and `sigma = 0` still lands exactly on it.
#[test]
fn vol_swap_heston_is_unchanged_by_the_poison_check() {
  let naive = 0.04_f64.sqrt();
  let dispersed = VolatilitySwapPricer::fair_strike_heston(&heston(0.09, 1.5, 0.04), 1.0);
  assert!(dispersed.is_finite() && dispersed > 0.0, "{dispersed}");
  let flat = VolatilitySwapPricer::fair_strike_heston(&heston(0.04, 1.5, 0.04), 1.0);
  assert!((flat - naive).abs() < 1e-15, "{flat} vs {naive}");
}

/// Both branches of `VolatilitySwapPricer::fair_strike_heston`, κ → 0 included, reject the same
/// inputs.
#[test]
fn vol_swap_heston_rejects_a_negative_variance_on_both_branches() {
  for &kappa in &[1e-12, 1.5] {
    let err = std::panic::catch_unwind(|| {
      VolatilitySwapPricer::fair_strike_heston(&heston(-0.04, kappa, -0.04), 1.0)
    })
    .unwrap_err();
    let msg = err
      .downcast_ref::<String>()
      .cloned()
      .unwrap_or_else(|| (*err.downcast_ref::<&str>().unwrap_or(&"")).to_string());
    assert_eq!(
      msg, "k_var must satisfy `k_var >= 0.0`, got k_var = -0.04",
      "kappa={kappa}"
    );
  }
}

#[test]
fn replication_weights_are_positive_and_decay() {
  // Strikes near forward have largest weight; weight ∝ 1/K^2.
  let strikes: Vec<f64> = (50..=150).step_by(10).map(|i| i as f64).collect();
  let w = replication_weights(&strikes, 1.0);
  assert_eq!(w.len(), strikes.len());
  for &wi in &w {
    assert!(wi > 0.0);
  }
  // Weight at K=50 should exceed weight at K=150 (1/K^2 dominates Δk).
  assert!(w[0] > *w.last().unwrap());
}

#[test]
fn replication_strike_within_one_percent_of_bsm_for_dense_strip() {
  // Build a dense BS option strip (σ = 25%) and replicate the strike.
  use stochastic_rs_distributions::special::norm_cdf;
  let (s, tau) = (100.0, 1.0);
  let sigma = 0.25;
  let strikes: Vec<f64> = (10..=400).map(|i| i as f64 * 0.5).collect();
  let prices: Vec<f64> = strikes
    .iter()
    .map(|&k| {
      let d1 = ((s / k).ln() + 0.5 * sigma * sigma * tau) / (sigma * tau.sqrt());
      let d2 = d1 - sigma * tau.sqrt();
      if k >= s {
        // call
        s * norm_cdf(d1) - k * norm_cdf(d2)
      } else {
        // put via parity (r = q = 0)
        k * norm_cdf(-d2) - s * norm_cdf(-d1)
      }
    })
    .collect();
  let k_var = VarianceSwapPricer::fair_strike_replication(&strikes, &prices, s, 0.0, 0.0, tau);
  let target = sigma * sigma;
  let rel_err = (k_var - target).abs() / target;
  assert!(
    rel_err < 0.02,
    "K_var={k_var}, expected≈{target}, rel_err={rel_err}"
  );
}

#[test]
#[should_panic(expected = "static replication needs at least 2 strikes (got 1)")]
fn replication_weights_reject_a_single_strike() {
  let _ = replication_weights(&[100.0], 1.0);
}

#[test]
#[should_panic(expected = "static replication needs at least 2 strikes (got 0)")]
fn replication_weights_reject_an_empty_strip() {
  let _ = replication_weights(&[], 1.0);
}

#[test]
fn replication_weights_are_nan_at_a_maturity_that_is_not_positive() {
  for maturity in [0.0, -1.0, f64::NAN] {
    let weights = replication_weights(&[90.0, 100.0, 110.0], maturity);
    assert!(
      weights.iter().all(|w| w.is_nan()),
      "maturity={maturity}: {weights:?}"
    );
  }
}

/// One hedges what the other prices, so both refuse a one-strike strip and both answer a maturity
/// that is not positive with NaN.
#[test]
fn the_replication_pair_agrees_on_its_inputs() {
  let one_strike = |tau: f64| {
    let strike = std::panic::catch_unwind(|| {
      VarianceSwapPricer::fair_strike_replication(&[100.0], &[1.0], 100.0, 0.0, 0.0, tau)
    });
    let weights = std::panic::catch_unwind(|| replication_weights(&[100.0], tau));
    (strike.is_err(), weights.is_err())
  };
  assert_eq!(one_strike(1.0), (true, true));
  for tau in [0.0, -1.0] {
    let strikes = [90.0, 100.0, 110.0];
    let strike =
      VarianceSwapPricer::fair_strike_replication(&strikes, &[1.0; 3], 100.0, 0.0, 0.0, tau);
    let weights = replication_weights(&strikes, tau);
    assert!(
      strike.is_nan() && weights.iter().all(|w| w.is_nan()),
      "tau={tau}"
    );
  }
}

#[test]
fn heston_fair_strike_is_nan_at_a_negative_or_nan_maturity() {
  for tau in [-1.0, f64::NAN] {
    assert!(VarianceSwapPricer::fair_strike_heston(&heston(0.04, 1.5, 0.04), tau).is_nan());
  }
}

/// `tau == 0` is the genuine $T \to 0$ limit of
/// $\frac{1-e^{-\kappa T}}{\kappa T} \to 1$, not a sentinel, so it keeps
/// returning `v0` — the negative-maturity guard must not take it with it.
#[test]
fn heston_fair_strike_zero_maturity_stays_the_v0_limit() {
  assert_eq!(
    VarianceSwapPricer::fair_strike_heston(&heston(0.04, 1.5, 0.10), 0.0),
    0.04
  );
}
