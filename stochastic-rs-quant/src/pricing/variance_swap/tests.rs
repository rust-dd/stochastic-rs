use super::*;

fn pricer() -> VarianceSwapPricer {
  VarianceSwapPricer {
    s: 100.0,
    r: 0.05,
    q: 0.0,
    tau: 1.0,
  }
}

#[test]
fn bsm_fair_strike_is_sigma_squared() {
  assert!((pricer().fair_strike_bsm(0.2) - 0.04).abs() < 1e-12);
}

#[test]
fn forward_under_zero_dividend() {
  let p = pricer();
  assert!((p.forward() - 100.0 * 0.05_f64.exp()).abs() < 1e-10);
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
  assert!((VarianceSwapPricer::pnl(0.06, 0.04, 100_000.0) - 2_000.0).abs() < 1e-9);
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
  let p = pricer();
  let k_var = p.fair_strike_heston(0.04, 1.5, 0.04);
  assert!((k_var - 0.04).abs() < 1e-12);
}

#[test]
fn heston_fair_strike_blends_v0_to_theta() {
  // V0 = 0.09 (high IV), θ = 0.04 (low LR), strong κ → fair K close to θ
  let p = pricer();
  let k_strong = p.fair_strike_heston(0.09, 5.0, 0.04);
  let k_weak = p.fair_strike_heston(0.09, 0.1, 0.04);
  assert!(k_strong < k_weak);
  assert!(k_weak <= 0.09);
  assert!(k_strong > 0.04);
}

#[test]
fn heston_kappa_zero_limit_equals_v0() {
  let p = pricer();
  assert!((p.fair_strike_heston(0.04, 0.0, 0.10) - 0.04).abs() < 1e-12);
}

#[test]
fn heston_long_t_limit_approaches_theta() {
  // T → ∞ with κ > 0 ⇒ factor → 0, K_var → θ.
  let p = VarianceSwapPricer {
    s: 100.0,
    r: 0.0,
    q: 0.0,
    tau: 50.0,
  };
  let k_var = p.fair_strike_heston(0.09, 2.0, 0.04);
  assert!(
    (k_var - 0.04).abs() < 0.01,
    "K_var={k_var} should approach θ=0.04"
  );
}

#[test]
fn heston_discrete_correction_vanishes_with_n() {
  let p = pricer();
  let k_cont = p.fair_strike_heston(0.04, 1.5, 0.04);
  let k_disc_fine = p.fair_strike_heston_discrete(0.04, 1.5, 0.04, 0.3, -0.7, 100_000);
  let k_disc_coarse = p.fair_strike_heston_discrete(0.04, 1.5, 0.04, 0.3, -0.7, 12);
  assert!((k_disc_fine - k_cont).abs() < (k_disc_coarse - k_cont).abs());
}

/// A non-positive maturity is invalid input, not a not-computable point:
/// there is no window to annualise over. It returned `0.0` before, which
/// `bsm_fair_strike_is_sigma_squared` shows is a value the same type also
/// produces as a genuine answer.
#[test]
#[should_panic(expected = "maturity tau must be strictly positive (got 0)")]
fn replication_rejects_a_nonpositive_maturity() {
  let p = VarianceSwapPricer {
    s: 100.0,
    r: 0.0,
    q: 0.0,
    tau: 0.0,
  };
  let _ = p.fair_strike_replication(&[90.0, 100.0, 110.0], &[1.0, 2.0, 1.0]);
}

/// The trapezoidal weights read `strikes[i±1]`, so one strike is not a
/// coarse strip — it is not a strip. Checked at both reachable lengths so
/// the guard cannot pass on the empty case alone.
#[test]
fn replication_rejects_a_strip_shorter_than_two_strikes() {
  for (strikes, prices) in [(&[][..], &[][..]), (&[100.0][..], &[2.0][..])] {
    let err =
      std::panic::catch_unwind(|| pricer().fair_strike_replication(strikes, prices)).unwrap_err();
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
  let p = pricer();
  let strikes = [90.0, 100.0, 110.0];
  let got = p.fair_strike_replication(&strikes, &[1.0, f64::NAN, 1.0]);
  assert!(
    got.is_nan(),
    "a NaN price must not floor to a strike, got {got}"
  );

  let clean = p.fair_strike_replication(&strikes, &[1.0, 2.0, 1.0]);
  assert!(
    clean.is_finite(),
    "control case must still price, got {clean}"
  );
}

/// "No observations" and "no movement" were the same number before —
/// `realized_variance_constant_path_is_zero` above is the genuine `0.0`
/// this one used to be indistinguishable from.
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

/// A negative variance is programmer error by the crate convention, and a
/// zero one leaves the Jensen correction dividing by `k_var^{3/2}`.
/// `vol_swap_zero_dispersion_recovers_sqrt_var` above is the real `0.2`
/// this used to collide with at the bottom of its range.
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

/// A `NaN` vol-of-vol is the one undefined input the `k_var > 0` guard
/// cannot see: `k_var` is built from `(v0, kappa, theta, tau)` and does not
/// read `sigma` at all, so the assertion passes and the `NaN` arrives at
/// the Jensen correction intact. `f64::NAN.max(0.0)` is `0.0`, so the floor
/// used to hand back `sqrt(k_var)` — exactly `0.2` here, which is the
/// number `vol_swap_zero_dispersion_recovers_sqrt_var` pins as the *real*
/// answer for a genuinely dispersion-free swap. The two were
/// indistinguishable.
#[test]
fn vol_swap_heston_preserves_a_nan_vol_of_vol() {
  let k = VolatilitySwapPricer::fair_strike_heston(0.04, 1.5, 0.04, f64::NAN, 1.0);
  assert!(k.is_nan(), "a NaN sigma must exit as NaN, got {k}");
}

/// The poison check must not disturb the dispersion it is guarding: a
/// finite vol-of-vol still lowers the strike below `sqrt(k_var)` by the
/// convexity correction, and `sigma = 0` still lands exactly on it.
#[test]
fn vol_swap_heston_is_unchanged_by_the_poison_check() {
  let naive = 0.04_f64.sqrt();
  let dispersed = VolatilitySwapPricer::fair_strike_heston(0.09, 1.5, 0.04, 0.3, 1.0);
  assert!(dispersed.is_finite() && dispersed > 0.0, "{dispersed}");
  let flat = VolatilitySwapPricer::fair_strike_heston(0.04, 1.5, 0.04, 0.3, 1.0);
  assert!((flat - naive).abs() < 1e-15, "{flat} vs {naive}");
}

/// Both branches of `VolatilitySwapPricer::fair_strike_heston` must reject
/// the same inputs. The κ → 0 short-circuit used to floor a negative
/// strike through `max(0.0)` and return `0.0` while the main path returned
/// its own sentinel, so a caller sweeping κ would have seen the guard
/// change shape underneath them.
#[test]
fn vol_swap_heston_rejects_a_negative_variance_on_both_branches() {
  for &kappa in &[1e-12, 1.5] {
    let err = std::panic::catch_unwind(|| {
      VolatilitySwapPricer::fair_strike_heston(-0.04, kappa, -0.04, 0.3, 1.0)
    })
    .unwrap_err();
    let msg = err
      .downcast_ref::<String>()
      .cloned()
      .unwrap_or_else(|| (*err.downcast_ref::<&str>().unwrap_or(&"")).to_string());
    assert!(
      msg.contains("heston variance strike must be strictly positive"),
      "kappa={kappa} gave the wrong panic: {msg}"
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
  let p = VarianceSwapPricer {
    s: 100.0,
    r: 0.0,
    q: 0.0,
    tau: 1.0,
  };
  let sigma = 0.25;
  let strikes: Vec<f64> = (10..=400).map(|i| i as f64 * 0.5).collect();
  let prices: Vec<f64> = strikes
    .iter()
    .map(|&k| {
      let d1 = ((p.s / k).ln() + 0.5 * sigma * sigma * p.tau) / (sigma * p.tau.sqrt());
      let d2 = d1 - sigma * p.tau.sqrt();
      if k >= p.s {
        // call
        p.s * norm_cdf(d1) - k * norm_cdf(d2)
      } else {
        // put via parity (r = q = 0)
        k * norm_cdf(-d2) - p.s * norm_cdf(-d1)
      }
    })
    .collect();
  let k_var = p.fair_strike_replication(&strikes, &prices);
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
#[should_panic(expected = "maturity must be strictly positive (got 0)")]
fn replication_weights_reject_a_zero_maturity() {
  let _ = replication_weights(&[90.0, 100.0, 110.0], 0.0);
}

#[test]
#[should_panic(expected = "maturity must be strictly positive (got -1)")]
fn replication_weights_reject_a_negative_maturity() {
  let _ = replication_weights(&[90.0, 100.0, 110.0], -1.0);
}

/// `replication_weights` and `fair_strike_replication` are meant to be used
/// together — one hedges what the other prices — so they must agree about
/// which inputs are errors. They did not: the sibling panicked on a
/// one-point strip and a non-positive maturity while this one returned an
/// all-zero weight vector, which replicates nothing and prices a zero
/// strike downstream with no signal.
#[test]
fn the_replication_pair_rejects_the_same_inputs() {
  for (strikes, tau) in [
    (vec![100.0], 1.0),
    (vec![90.0, 100.0, 110.0], 0.0),
    (vec![90.0, 100.0, 110.0], -1.0),
  ] {
    let prices = vec![1.0; strikes.len()];
    let p = VarianceSwapPricer {
      s: 100.0,
      r: 0.0,
      q: 0.0,
      tau,
    };
    let sibling =
      std::panic::catch_unwind(|| p.fair_strike_replication(&strikes, &prices)).is_err();
    let weights = std::panic::catch_unwind(|| replication_weights(&strikes, tau)).is_err();
    assert_eq!(
      sibling, weights,
      "strikes={strikes:?} tau={tau}: fair_strike_replication panicked={sibling}, \
       replication_weights panicked={weights}"
    );
    assert!(sibling, "strikes={strikes:?} tau={tau} must be rejected");
  }
}

#[test]
#[should_panic(expected = "maturity tau must be non-negative (got -1)")]
fn heston_fair_strike_rejects_a_negative_maturity() {
  let p = VarianceSwapPricer {
    s: 100.0,
    r: 0.05,
    q: 0.0,
    tau: -1.0,
  };
  let _ = p.fair_strike_heston(0.04, 1.5, 0.04);
}

/// `tau == 0` is the genuine $T \to 0$ limit of
/// $\frac{1-e^{-\kappa T}}{\kappa T} \to 1$, not a sentinel, so it keeps
/// returning `v0` — the negative-maturity guard must not take it with it.
#[test]
fn heston_fair_strike_zero_maturity_stays_the_v0_limit() {
  let p = VarianceSwapPricer {
    s: 100.0,
    r: 0.05,
    q: 0.0,
    tau: 0.0,
  };
  assert_eq!(p.fair_strike_heston(0.04, 1.5, 0.10), 0.04);
}
