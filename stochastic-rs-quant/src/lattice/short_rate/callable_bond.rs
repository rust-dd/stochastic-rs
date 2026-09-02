//! Callable and puttable coupon bonds on the one-factor short-rate trees.
//!
//! Backward induction through the trinomial lattice with the bond's coupons
//! as intermediate cash flows and the embedded options exercised at their
//! decision nodes: the issuer calls when the ex-coupon continuation value
//! exceeds the call price (`V ← min(V, K_call)`), the holder puts when it
//! falls below the put price (`V ← max(V, K_put)`), and the coupon due at
//! that date is added afterwards. Exercise prices are clean. Coupon and
//! exercise times snap to the nearest tree level, so the step count should
//! make the grid spacing divide the coupon period.
//!
//! References: Hull, J. C. (2018), *Options, Futures, and Other Derivatives*,
//! 10th ed., Pearson, §31.5–31.6 (Hull–White trees, bonds with embedded
//! options); Hull, J. & White, A. (1994), *Numerical Procedures for
//! Implementing Term Structure Models I: Single-Factor Models*, Journal of
//! Derivatives 2(1), 7–16.

use ndarray::Array1;

use super::OneFactorShortRateModel;
use crate::lattice::tree::TrinomialTree;
use crate::traits::RealExt;

/// Coupon bond with optional call and put schedules; times are year
/// fractions from the valuation date.
#[derive(Debug, Clone)]
pub struct CallableBondSpec<T: RealExt> {
  /// Face value redeemed at maturity.
  pub face: T,
  /// Annual coupon rate; the coupon paid at `t_i` is `face · rate · (t_i − t_{i−1})`.
  pub coupon_rate: T,
  /// Increasing coupon payment times, the last one being the maturity.
  pub coupon_times: Vec<T>,
  /// `(time, clean price)` pairs at which the issuer may call.
  pub call_schedule: Vec<(T, T)>,
  /// `(time, clean price)` pairs at which the holder may put.
  pub put_schedule: Vec<(T, T)>,
}

impl<T: RealExt> CallableBondSpec<T> {
  /// Straight coupon bond; add the embedded options with
  /// [`with_calls`](Self::with_calls) and [`with_puts`](Self::with_puts).
  pub fn new(face: T, coupon_rate: T, coupon_times: Vec<T>) -> Self {
    assert!(
      !coupon_times.is_empty(),
      "a bond needs at least one coupon date"
    );
    assert!(
      coupon_times.windows(2).all(|w| w[0] < w[1]) && coupon_times[0] > T::zero(),
      "coupon times must be positive and increasing"
    );
    Self {
      face,
      coupon_rate,
      coupon_times,
      call_schedule: Vec::new(),
      put_schedule: Vec::new(),
    }
  }

  /// Adds the issuer's call schedule.
  pub fn with_calls(mut self, schedule: Vec<(T, T)>) -> Self {
    self.call_schedule = schedule;
    self
  }

  /// Adds the holder's put schedule.
  pub fn with_puts(mut self, schedule: Vec<(T, T)>) -> Self {
    self.put_schedule = schedule;
    self
  }

  /// Maturity, the last coupon time.
  pub fn maturity(&self) -> T {
    *self.coupon_times.last().expect("non-empty by construction")
  }
}

/// Bond prices with and without the embedded options. `call_value` is the
/// straight price minus the price with the calls alone, `put_value` the price
/// with the puts alone minus the straight price; `price` carries both
/// schedules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CallableBondPrice<T> {
  pub price: T,
  pub straight_price: T,
  pub call_value: T,
  pub put_value: T,
}

/// Prices the bond on a one-factor short-rate tree whose horizon is the
/// bond's maturity.
pub fn price_callable_bond<T: RealExt, M: OneFactorShortRateModel<T>>(
  tree: &TrinomialTree<T>,
  model: &M,
  spec: &CallableBondSpec<T>,
) -> CallableBondPrice<T> {
  let straight = sweep(tree, model, spec, false, false);
  let has_calls = !spec.call_schedule.is_empty();
  let has_puts = !spec.put_schedule.is_empty();
  let with_calls = if has_calls {
    sweep(tree, model, spec, true, false)
  } else {
    straight
  };
  let with_puts = if has_puts {
    sweep(tree, model, spec, false, true)
  } else {
    straight
  };
  let price = match (has_calls, has_puts) {
    (false, false) => straight,
    (true, false) => with_calls,
    (false, true) => with_puts,
    (true, true) => sweep(tree, model, spec, true, true),
  };
  CallableBondPrice {
    price,
    straight_price: straight,
    call_value: straight - with_calls,
    put_value: with_puts - straight,
  }
}

/// Tree level nearest to `time`.
fn level_of<T: RealExt>(time: T, dt: T, last: usize) -> usize {
  let raw = (time / dt).round().to_f64().unwrap_or(0.0).max(0.0) as usize;
  raw.min(last)
}

fn sweep<T: RealExt, M: OneFactorShortRateModel<T>>(
  tree: &TrinomialTree<T>,
  model: &M,
  spec: &CallableBondSpec<T>,
  calls: bool,
  puts: bool,
) -> T {
  let last = tree.states.len() - 1;
  let dt = tree.dt;
  let horizon = T::from_usize_(last) * dt;
  assert!(
    (spec.maturity() - horizon).abs() <= dt / T::from_f64_fast(2.0),
    "the bond's maturity must sit on the tree horizon"
  );
  let mut coupon = vec![T::zero(); last + 1];
  let mut prev = T::zero();
  for &t in &spec.coupon_times {
    coupon[level_of(t, dt, last)] += spec.face * spec.coupon_rate * (t - prev);
    prev = t;
  }
  let mut call_at: Vec<Option<T>> = vec![None; last + 1];
  if calls {
    for &(t, k) in &spec.call_schedule {
      call_at[level_of(t, dt, last)] = Some(k);
    }
  }
  let mut put_at: Vec<Option<T>> = vec![None; last + 1];
  if puts {
    for &(t, k) in &spec.put_schedule {
      put_at[level_of(t, dt, last)] = Some(k);
    }
  }
  let mut values = Array1::from_elem(tree.states[last].len(), spec.face + coupon[last]);
  for level in (0..last).rev() {
    let width = tree.states[level].len();
    let time = T::from_usize_(level) * dt;
    let mut next = Array1::<T>::zeros(width);
    for node in 0..width {
      let branch = tree.branches[level][node];
      let expected = branch.down_probability * values[branch.center_index - 1]
        + branch.middle_probability * values[branch.center_index]
        + branch.up_probability * values[branch.center_index + 1];
      let mut value = (-model.short_rate(time, tree.states[level][node]) * dt).exp() * expected;
      if let Some(k) = call_at[level] {
        value = value.min(k);
      }
      if let Some(k) = put_at[level] {
        value = value.max(k);
      }
      next[node] = value + coupon[level];
    }
    values = next;
  }
  values[0]
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bonds::Vasicek;
  use crate::lattice::short_rate::HullWhiteTree;
  use crate::lattice::short_rate::HullWhiteTreeModel;
  use crate::traits::ShortRatePricer;

  fn model() -> HullWhiteTreeModel<f64> {
    HullWhiteTreeModel::new(0.04, 0.3, 0.04, 0.01)
  }

  fn annual_bond(years: usize, coupon: f64) -> CallableBondSpec<f64> {
    CallableBondSpec::new(100.0, coupon, (1..=years).map(|i| i as f64).collect())
  }

  /// Without options the sweep is linear in the cash flows, so the price is
  /// the coupon-weighted sum of the tree's own zero-coupon bonds — trees
  /// sharing `Δt` share their nodes and branch probabilities.
  #[test]
  fn straight_bond_is_the_sum_of_tree_zero_coupon_bonds() {
    let tree = HullWhiteTree::new(model(), 3.0, 36);
    let priced = price_callable_bond(&tree.tree, &tree.model, &annual_bond(3, 0.05));
    let zcb =
      |years: usize| HullWhiteTree::new(model(), years as f64, 12 * years).zero_coupon_bond_price();
    let want = 5.0 * (zcb(1) + zcb(2) + zcb(3)) + 100.0 * zcb(3);
    assert!(
      (priced.straight_price - want).abs() < 1e-9,
      "{} vs {want}",
      priced.straight_price
    );
    assert_eq!(priced.price, priced.straight_price);
    assert_eq!(priced.call_value, 0.0);
    assert_eq!(priced.put_value, 0.0);
  }

  /// The tree model is Vasicek with constant θ, so the straight bond must
  /// converge to the closed-form coupon bond.
  #[test]
  fn straight_bond_converges_to_the_vasicek_closed_form() {
    let tree = HullWhiteTree::new(model(), 3.0, 360);
    let price = price_callable_bond(&tree.tree, &tree.model, &annual_bond(3, 0.05)).straight_price;
    let vasicek = Vasicek {
      theta: 0.3,
      mu: 0.04,
      sigma: 0.01,
    };
    let want = (1..=3)
      .map(|i| 5.0 * vasicek.zero_coupon_price(0.04, i as f64))
      .sum::<f64>()
      + 100.0 * vasicek.zero_coupon_price(0.04, 3.0);
    assert!((price - want).abs() / want < 3e-3, "{price} vs {want}");
  }

  #[test]
  fn embedded_options_move_the_price_in_the_expected_direction() {
    let tree = HullWhiteTree::new(model(), 3.0, 36);
    let straight = annual_bond(3, 0.06);
    let price = |spec: &CallableBondSpec<f64>| price_callable_bond(&tree.tree, &tree.model, spec);
    let s = price(&straight).price;
    let callable = price(
      &straight
        .clone()
        .with_calls(vec![(1.0, 100.0), (2.0, 100.0)]),
    );
    let puttable = price(&straight.clone().with_puts(vec![(1.0, 100.0), (2.0, 100.0)]));
    let both = price(
      &straight
        .clone()
        .with_calls(vec![(1.0, 100.0), (2.0, 100.0)])
        .with_puts(vec![(1.0, 100.0), (2.0, 100.0)]),
    );
    assert!(callable.price < s && callable.call_value > 0.0);
    assert!((callable.call_value - (s - callable.price)).abs() < 1e-12);
    assert!(puttable.price > s && puttable.put_value > 0.0);
    assert!(both.price >= callable.price && both.price <= puttable.price);
    let far = price(&straight.clone().with_calls(vec![(1.0, 1.0e6)]));
    assert_eq!(far.price, s);
  }

  /// With a vanishing volatility and θ = r₀ the short rate is flat at 2 %, so
  /// a 5 % coupon trades above par and is called at par on the first call
  /// date: the holder receives the coupon plus par at `t = 1`.
  #[test]
  fn deterministic_rates_call_the_premium_bond_at_the_first_date() {
    let model = HullWhiteTreeModel::new(0.02, 0.3, 0.02, 1e-9);
    let tree = HullWhiteTree::new(model.clone(), 3.0, 36);
    let spec = annual_bond(3, 0.05).with_calls(vec![(1.0, 100.0), (2.0, 100.0)]);
    let priced = price_callable_bond(&tree.tree, &model, &spec);
    let want = 105.0 * (-0.02_f64).exp();
    assert!(
      (priced.price - want).abs() < 1e-6,
      "{} vs {want}",
      priced.price
    );
    assert!(priced.straight_price > priced.price);
  }

  #[test]
  #[should_panic(expected = "maturity must sit on the tree horizon")]
  fn rejects_a_maturity_off_the_horizon() {
    let tree = HullWhiteTree::new(model(), 3.0, 36);
    let _ = price_callable_bond(&tree.tree, &tree.model, &annual_bond(2, 0.05));
  }
}
