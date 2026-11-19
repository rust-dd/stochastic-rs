//! Hagan's CMS formulas against mpmath at 50 digits, and the edges of the valuation.

use super::*;
use crate::instruments::option::types::VolatilityQuoteKind;
use crate::instruments::option::volatility::BlackVolatility;
use crate::instruments::option::volatility::SabrVolatility;

/// A vol that falls with the strike, of either quote family.
#[derive(Debug, Clone, Copy)]
struct Skew {
  atm: f64,
  slope: f64,
  kind: VolatilityQuoteKind,
}

impl VolatilityModel<f64> for Skew {
  fn implied_volatility(&self, forward: f64, strike: f64, _tau: f64) -> f64 {
    self.atm - self.slope * (strike - forward)
  }

  fn quote_kind(&self) -> VolatilityQuoteKind {
    self.kind
  }
}

fn flat(sigma: f64) -> Skew {
  Skew {
    atm: sigma,
    slope: 0.0,
    kind: VolatilityQuoteKind::Lognormal,
  }
}

fn normal(sigma: f64) -> Skew {
  Skew {
    atm: sigma,
    slope: 0.0,
    kind: VolatilityQuoteKind::Normal,
  }
}

/// `(swap_years, fixed_freq, payment_delay)` of the referenced swap.
type Swap = (f64, Frequency, f64);

const TEN_SEMI: Swap = (10.0, Frequency::SemiAnnual, 0.0);

/// The caplet's and the floorlet's valuations per unit of notional, accrual and discount.
fn pair<V: VolatilityModel<f64> + Copy>(
  s0: f64,
  strike: f64,
  t_fix: f64,
  swap: Swap,
  vol: V,
) -> (CmsValuation<f64>, CmsValuation<f64>) {
  let caplet = CmsCaplet {
    strike,
    notional: 1.0,
    accrual_factor: 1.0,
    discount_factor: 1.0,
    forward_cms: s0,
    t_fix,
    swap_years: swap.0,
    fixed_freq: swap.1,
    payment_delay: swap.2,
    vol,
  };
  let floorlet = CmsFloorlet {
    strike,
    notional: 1.0,
    accrual_factor: 1.0,
    discount_factor: 1.0,
    forward_cms: s0,
    t_fix,
    swap_years: swap.0,
    fixed_freq: swap.1,
    payment_delay: swap.2,
    vol,
  };
  (caplet.valuation(), floorlet.valuation())
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
  assert!(
    (got - want).abs() <= rel * want.abs(),
    "{what}: got {got:e}, want {want:e}"
  );
}

#[test]
fn a_known_fixing_pays_its_intrinsic_value_without_the_vol_model() {
  let vol = SabrVolatility::new(0.035_f64, 0.5, -0.2, 0.4);
  let caplet = |t_fix: f64| {
    CmsCaplet {
      strike: 0.02,
      notional: 1e6,
      accrual_factor: 1.0,
      discount_factor: 0.97,
      forward_cms: 0.03,
      t_fix,
      swap_years: 10.0,
      fixed_freq: Frequency::SemiAnnual,
      payment_delay: 0.0,
      vol,
    }
    .npv()
  };
  let floorlet = |t_fix: f64| {
    CmsFloorlet {
      strike: 0.04,
      notional: 1e6,
      accrual_factor: 1.0,
      discount_factor: 0.97,
      forward_cms: 0.03,
      t_fix,
      swap_years: 10.0,
      fixed_freq: Frequency::SemiAnnual,
      payment_delay: 0.0,
      vol,
    }
    .npv()
  };
  for t_fix in [-0.5, 0.0] {
    assert!((caplet(t_fix) - 9700.0).abs() < 1e-6, "t_fix = {t_fix}");
    assert!((floorlet(t_fix) - 9700.0).abs() < 1e-6, "t_fix = {t_fix}");
  }
  assert!(caplet(0.5).is_finite() && caplet(0.5) != caplet(0.0));
}

/// Hagan's (3.5b) over `D(t_p)`, less the forward; the last column is mpmath's.
#[test]
fn the_lognormal_adjustment_is_hagans_3_5b() {
  let semi = Frequency::SemiAnnual;
  let rows = [
    (0.04, 0.3, 2.0, 10.0, semi, 0.0, 0.001522561643760736),
    (0.04, 0.3, 2.0, 10.0, semi, 0.5, 0.0013678813589593163),
    (0.04, 0.2, 5.0, 10.0, semi, 0.0, 0.0017092782403205616),
    (
      0.03,
      0.4,
      10.0,
      30.0,
      Frequency::Annual,
      0.0,
      0.045988222958304989,
    ),
    (0.04, 0.3, 2.0, 1.0, semi, 0.0, 0.00023125468321796416),
    (
      0.01,
      0.5,
      1.0,
      5.0,
      Frequency::Quarterly,
      0.25,
      6.669980775910413e-5,
    ),
    (1e-6, 0.3, 2.0, 10.0, semi, 0.0, 1.0353889993260826e-12),
    (-0.005, 0.3, 2.0, 10.0, semi, 0.25, 2.491963864096852e-5),
  ];
  for (s0, sigma, t_fix, years, freq, delay, want) in rows {
    let got = hagan_linear_tsr_convexity_adjustment(s0, sigma, t_fix, years, freq, delay);
    close(got, want, 1e-12, &format!("S0 = {s0}, delay = {delay}"));
  }
}

/// The whole `G` of (2.13a) under (2.9), by mpmath quadrature: the first order is 8 % under it.
#[test]
fn the_first_order_adjustment_is_near_the_whole_g() {
  let got = hagan_linear_tsr_convexity_adjustment(0.04, 0.3, 2.0, 10.0, Frequency::SemiAnnual, 0.0);
  let whole = 0.001652391696014858;
  assert!(got < whole && got > 0.9 * whole, "{got:e}");
}

/// Hagan's (3.5c) and (3.5d) over `D(t_p)` at a flat vol; the last two columns are mpmath's.
#[test]
fn the_lognormal_caplet_and_floorlet_are_hagans_3_5c_and_3_5d() {
  let annual = (30.0, Frequency::Annual, 0.0);
  let delayed = (10.0, Frequency::SemiAnnual, 0.25);
  let rows = [
    (
      0.04,
      0.05,
      0.3,
      2.0,
      delayed,
      0.004229719178647897,
      0.01278449767728787,
    ),
    (
      0.04,
      0.03,
      0.3,
      2.0,
      TEN_SEMI,
      0.013467905368559095,
      0.0019453437247983587,
    ),
    (
      0.04,
      0.04,
      0.3,
      2.0,
      TEN_SEMI,
      0.007733057731602908,
      0.006210496087842172,
    ),
    (
      0.04,
      0.08,
      0.3,
      2.0,
      TEN_SEMI,
      0.0006832864877673029,
      0.03916072484400657,
    ),
    (
      0.03,
      0.035,
      0.4,
      10.0,
      annual,
      0.05394189371265165,
      0.012953670754346662,
    ),
  ];
  for (s0, strike, sigma, t_fix, swap, caplet, floorlet) in rows {
    let (c, p) = pair(s0, strike, t_fix, swap, flat(sigma));
    close(c.npv, caplet, 1e-12, &format!("caplet at K = {strike}"));
    close(p.npv, floorlet, 1e-12, &format!("floorlet at K = {strike}"));
  }
}

/// Hagan's (3.6b) to (3.6d), where the forward may be zero or negative; mpmath's columns.
#[test]
fn a_normal_vol_prices_by_hagans_3_6b_to_3_6d() {
  let delayed = (10.0, Frequency::SemiAnnual, 0.25);
  let five = (5.0, Frequency::Annual, 0.0);
  let rows = [
    (
      0.04,
      0.05,
      0.012,
      2.0,
      delayed,
      0.0013190515587825339,
      0.0032792967066890631,
      0.011960245147906529,
    ),
    (
      0.04,
      0.03,
      0.012,
      2.0,
      TEN_SEMI,
      0.0013896397940766516,
      0.013916340435860145,
      0.0025267006417834932,
    ),
    (
      -0.002,
      0.001,
      0.006,
      3.0,
      five,
      0.00032508259701985524,
      0.0029431039495012014,
      0.005618021352481346,
    ),
    (
      0.0,
      0.002,
      0.006,
      3.0,
      five,
      0.000324,
      0.0033597473065004815,
      0.005035747306500481,
    ),
  ];
  for (s0, strike, sigma, t_fix, swap, adjustment, caplet, floorlet) in rows {
    let (c, p) = pair(s0, strike, t_fix, swap, normal(sigma));
    close(
      c.convexity_adjustment,
      adjustment,
      1e-12,
      &format!("adjustment at S0 = {s0}"),
    );
    close(c.npv, caplet, 1e-12, &format!("caplet at S0 = {s0}"));
    close(p.npv, floorlet, 1e-12, &format!("floorlet at S0 = {s0}"));
  }
}

/// (2.21b): under a smile the caplet less the floorlet is one swaplet at every strike, since the
/// option in the money is priced as the swaplet at the ATM vol plus the other option.
#[test]
fn caplet_less_floorlet_is_one_swaplet_at_every_strike_under_a_smile() {
  for kind in [VolatilityQuoteKind::Lognormal, VolatilityQuoteKind::Normal] {
    let (atm, slope) = match kind {
      VolatilityQuoteKind::Lognormal => (0.3, 2.0),
      VolatilityQuoteKind::Normal => (0.012, 0.1),
    };
    let vol = Skew { atm, slope, kind };
    let swaplet = pair(0.04, 0.04, 2.0, TEN_SEMI, vol).0.convexity_adjustment;
    assert!(swaplet > 1e-3, "{kind}: {swaplet:e}");
    for strike in [0.01, 0.03, 0.04, 0.05, 0.09] {
      let (c, p) = pair(0.04, strike, 2.0, TEN_SEMI, vol);
      assert_eq!(c.convexity_adjustment, swaplet, "{kind}, K = {strike}");
      assert_eq!(p.convexity_adjustment, swaplet, "{kind}, K = {strike}");
      let gap = c.npv - p.npv - (0.04 + swaplet - strike);
      assert!(gap.abs() < 1e-15, "{kind}, K = {strike}: {gap:e}");
    }
  }
}

/// The Black caplet's reading of a forward at or below 0: the payoff on the adjusted forward where
/// the strike is across 0, NaN for two negative rates; the adjustment column is mpmath's.
#[test]
fn a_lognormal_vol_on_a_forward_at_or_below_zero_reads_it_as_the_black_caplet_does() {
  let rows = [
    (0.0, 0.02, 0.0, 0.0, 0.02),
    (
      -0.01,
      0.02,
      1.0571087557228397e-4,
      0.0,
      0.029894289124427716,
    ),
    (-0.01, 0.0, 1.0571087557228397e-4, 0.0, 0.009894289124427716),
    (0.0, -0.01, 0.0, 0.01, 0.0),
  ];
  for (s0, strike, adjustment, caplet, floorlet) in rows {
    let (c, p) = pair(s0, strike, 2.0, TEN_SEMI, flat(0.3));
    let row = format!("S0 = {s0}, K = {strike}");
    close(c.convexity_adjustment, adjustment, 1e-13, &row);
    close(c.npv, caplet, 1e-15, &row);
    close(p.npv, floorlet, 1e-15, &row);
  }
  let (c, p) = pair(-0.01, -0.02, 2.0, TEN_SEMI, flat(0.3));
  assert!(c.npv.is_nan() && p.npv.is_nan(), "{c:?} {p:?}");
}

#[test]
fn a_non_positive_strike_on_a_positive_forward_always_pays() {
  let adjustment = 0.001522561643760736;
  for strike in [0.0, -0.01] {
    let (c, p) = pair(0.04, strike, 2.0, TEN_SEMI, flat(0.3));
    close(
      c.npv,
      0.04 + adjustment - strike,
      1e-13,
      &format!("K = {strike}"),
    );
    assert_eq!(p.npv, 0.0, "K = {strike}");
  }
}

#[test]
fn a_zero_vol_pays_the_intrinsic_value_and_a_nan_vol_is_nan() {
  for vol in [flat(0.0), normal(0.0)] {
    let (c, p) = pair(0.04, 0.03, 2.0, TEN_SEMI, vol);
    assert_eq!(c.convexity_adjustment, 0.0);
    assert!((c.npv - 0.01).abs() < 1e-17 && p.npv == 0.0, "{c:?} {p:?}");
  }
  for vol in [flat(f64::NAN), normal(f64::NAN)] {
    let (c, p) = pair(0.04, 0.03, 2.0, TEN_SEMI, vol);
    assert!(c.npv.is_nan() && p.npv.is_nan() && c.convexity_adjustment.is_nan());
  }
  let black = pair(0.04, 0.03, 2.0, TEN_SEMI, BlackVolatility::new(0.3)).0;
  assert_eq!(black, pair(0.04, 0.03, 2.0, TEN_SEMI, flat(0.3)).0);
}

/// `S₀G'(S₀)/G(S₀)` of (2.13a) by mpmath, and its series where the closed form cancels.
#[test]
fn the_convexity_factor_is_the_log_slope_of_hagans_g() {
  let semi = Frequency::SemiAnnual;
  let rows = [
    (0.04, 10.0, semi, 0.0, 0.1930055269550905, 1e-14),
    (0.04, 10.0, semi, 0.5, 0.1733976838178356, 1e-14),
    (
      0.03,
      30.0,
      Frequency::Annual,
      0.0,
      0.3877885634877889,
      1e-14,
    ),
    (0.04, 1.0, semi, 0.0, 0.029314696175499903, 1e-13),
    (
      0.01,
      5.0,
      Frequency::Quarterly,
      0.25,
      0.023483746115733764,
      1e-13,
    ),
    (1e-6, 10.0, semi, 0.0, 5.249989062507547e-6, 1e-14),
    (-0.005, 10.0, semi, 0.25, -0.025271242091982589, 1e-13),
  ];
  for (s0, years, freq, delay, want, rel) in rows {
    let got = hagan_linear_tsr_convexity_factor(s0, years, freq, delay);
    close(got, want, rel, &format!("S0 = {s0}, {years}y"));
  }
  assert_eq!(hagan_linear_tsr_convexity_factor(0.0, 10.0, semi, 0.0), 0.0);
  for (s0, years) in [
    (-2.5, 10.0),
    (-2.0, 10.0),
    (0.04, 0.0),
    (0.04, -1.0),
    (f64::NAN, 10.0),
  ] {
    let factor = hagan_linear_tsr_convexity_factor(s0, years, semi, 0.0);
    assert!(factor.is_nan(), "S0 = {s0}, {years}y: {factor}");
  }
  // Either side of the switch at `n S δ = 1e-3`, per unit of rate.
  let per_rate = |s0: f64| hagan_linear_tsr_convexity_factor(s0, 10.0, semi, 0.0) / s0;
  let (series, closed) = (per_rate(0.999_999_9e-4), per_rate(1.000_000_1e-4));
  assert!((series - closed).abs() < 1e-9 * closed, "{series} {closed}");
}
