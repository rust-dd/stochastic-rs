use ndarray::array;

use super::*;

fn iid_basket(
  n: usize,
  sigma: f64,
  rho: f64,
) -> (
  Array1<f64>,
  Array1<f64>,
  Array1<f64>,
  Array1<f64>,
  Array2<f64>,
) {
  let s = Array1::from_elem(n, 100.0);
  let w = Array1::from_elem(n, 1.0 / n as f64);
  let sig = Array1::from_elem(n, sigma);
  let q = Array1::from_elem(n, 0.0);
  let mut rho_m = Array2::<f64>::from_elem((n, n), rho);
  for i in 0..n {
    rho_m[[i, i]] = 1.0;
  }
  (s, w, sig, q, rho_m)
}

/// Cross-arch tolerance: these goldens come from `norm_cdf`, whose last bit
/// is a hostage to FMA contraction and libm differences between the
/// aarch64-darwin dev machine and CI's ubuntu x86_64.
const TOL: f64 = 1e-12;

/// An asymmetric basket: distinct weights, spots, volatilities and dividend
/// yields, an indefinite-looking off-diagonal correlation, a non-round
/// strike and a non-unit maturity. Nothing here can survive a query field
/// left behind on the struct by coinciding with a default.
fn asymmetric_basket() -> (
  Array1<f64>,
  Array1<f64>,
  Array1<f64>,
  Array1<f64>,
  Array2<f64>,
) {
  (
    array![95.0, 110.0, 88.0],
    array![0.5, 0.3, 0.2],
    array![0.21, 0.34, 0.17],
    array![0.01, 0.02, 0.0],
    array![[1.0, 0.35, -0.2], [0.35, 1.0, 0.11], [-0.2, 0.11, 1.0]],
  )
}

/// Values captured from the bundled-market-data `GeometricBasketPricer`
/// **before** the model/query reshape. The reshape is an API change only,
/// so these must not move.
#[test]
fn geometric_basket_matches_pre_refactor_goldens() {
  let single = GeometricBasketPricer::new(array![1.0], array![0.2], array![[1.0]]);
  let s = array![100.0];
  let q = array![0.0];
  let call = single.price_call(s.view(), 100.0, 0.05, q.view(), 1.0);
  assert!((call - 10.450575415435111).abs() < TOL, "n=1 call {call}");
  let put = single.price_put(s.view(), 100.0, 0.05, q.view(), 1.0);
  assert!((put - 5.573517865506494).abs() < TOL, "n=1 put {put}");

  let (s, w, sig, q, rho) = asymmetric_basket();
  let model = GeometricBasketPricer::new(w, sig, rho);
  let call = model.price_call(s.view(), 97.5, 0.031, q.view(), 1.4);
  assert!((call - 7.9321080037610905).abs() < TOL, "asym call {call}");
  let put = model.price_put(s.view(), 97.5, 0.031, q.view(), 1.4);
  assert!((put - 7.226234312037958).abs() < TOL, "asym put {put}");
}

/// One model instance prices a whole strike and maturity grid — the point
/// of the split. A strike or maturity cached at construction would return
/// the same number three times.
#[test]
fn geometric_basket_one_model_prices_a_grid() {
  let (s, w, sig, q, rho) = asymmetric_basket();
  let model = GeometricBasketPricer::new(w, sig, rho);
  let strikes = [80.0, 97.5, 120.0].map(|k| model.price_call(s.view(), k, 0.031, q.view(), 1.4));
  assert!(
    strikes[0] > strikes[1] && strikes[1] > strikes[2],
    "basket calls must decay in the strike: {strikes:?}"
  );
  let taus = [0.25, 1.4, 4.0].map(|tau| model.price_call(s.view(), 97.5, 0.031, q.view(), tau));
  assert!(
    taus[0] < taus[1] && taus[1] < taus[2],
    "basket calls must rise in tau: {taus:?}"
  );
}

/// Geometric basket on a single asset must equal a vanilla BSM call.
#[test]
fn geometric_basket_n1_matches_bsm() {
  let p = GeometricBasketPricer::new(array![1.0], array![0.2], array![[1.0]]);
  let s = array![100.0];
  let q = array![0.0];
  let price = p.price_call(s.view(), 100.0, 0.05, q.view(), 1.0);
  let bsm = 10.4506;
  assert!((price - bsm).abs() < 0.005, "geo n=1: {price}");
}

/// Geometric basket with perfectly correlated identical assets equals
/// a single-asset vanilla.
#[test]
fn geometric_basket_perfect_corr_equals_single() {
  let (s, w, sig, q, rho) = iid_basket(5, 0.20, 1.0);
  let p = GeometricBasketPricer::new(w, sig, rho);
  let price = p.price_call(s.view(), 100.0, 0.05, q.view(), 1.0);
  let bsm = 10.4506;
  assert!((price - bsm).abs() < 0.01, "geo perf-corr: {price}");
}

/// Geometric basket should be cheaper than arithmetic (Jensen).
#[test]
fn geometric_below_arithmetic() {
  let (s, w, sig, q, rho) = iid_basket(4, 0.30, 0.5);
  let geo = GeometricBasketPricer::new(w.clone(), sig.clone(), rho.clone()).price_call(
    s.view(),
    100.0,
    0.04,
    q.view(),
    1.0,
  );
  let ari =
    ArithmeticBasketLevyPricer::new(w, sig, rho).price_call(s.view(), 100.0, 0.04, q.view(), 1.0);
  assert!(geo < ari, "geo={geo} should be < ari={ari}");
}

/// Values captured from the bundled-market-data
/// `ArithmeticBasketLevyPricer` **before** the model/query reshape. The
/// reshape is an API change only, so these must not move.
#[test]
fn arithmetic_basket_levy_matches_pre_refactor_goldens() {
  let (s, w, sig, q, rho) = asymmetric_basket();
  let model = ArithmeticBasketLevyPricer::new(w, sig, rho);
  let call = model.price_call(s.view(), 97.5, 0.031, q.view(), 1.4);
  assert!((call - 9.750478538898065).abs() < TOL, "asym call {call}");
  let put = model.price_put(s.view(), 97.5, 0.031, q.view(), 1.4);
  assert!((put - 6.581038540006583).abs() < TOL, "asym put {put}");

  let (s, w, sig, q, rho) = iid_basket(4, 0.30, 0.5);
  let model = ArithmeticBasketLevyPricer::new(w, sig, rho);
  let call = model.price_call(s.view(), 100.0, 0.04, q.view(), 1.0);
  assert!((call - 11.361162412209948).abs() < TOL, "iid call {call}");
  let put = model.price_put(s.view(), 100.0, 0.04, q.view(), 1.0);
  assert!((put - 7.440106327442262).abs() < TOL, "iid put {put}");
}

/// One model instance prices a whole strike grid — the point of the split.
#[test]
fn arithmetic_basket_levy_one_model_prices_a_strike_grid() {
  let (s, w, sig, q, rho) = asymmetric_basket();
  let model = ArithmeticBasketLevyPricer::new(w, sig, rho);
  let calls = [80.0, 97.5, 120.0].map(|k| model.price_call(s.view(), k, 0.031, q.view(), 1.4));
  let puts = [80.0, 97.5, 120.0].map(|k| model.price_put(s.view(), k, 0.031, q.view(), 1.4));
  assert!(
    calls[0] > calls[1] && calls[1] > calls[2],
    "basket calls must decay in the strike: {calls:?}"
  );
  assert!(
    puts[0] < puts[1] && puts[1] < puts[2],
    "basket puts must rise in the strike: {puts:?}"
  );
}

/// Levy and MC should agree within ~3% for a 4-asset arithmetic basket.
#[cfg(feature = "openblas")]
#[test]
fn levy_vs_mc_arithmetic() {
  let (s, w, sig, q, rho) = iid_basket(4, 0.25, 0.4);
  let levy = ArithmeticBasketLevyPricer::new(w.clone(), sig.clone(), rho.clone()).price_call(
    s.view(),
    100.0,
    0.05,
    q.view(),
    1.0,
  );
  let mc = McBasketPricer::new(w, BasketAverageType::Arithmetic, sig, rho, 100_000).price_call(
    s.view(),
    100.0,
    0.05,
    q.view(),
    1.0,
  );
  let rel = (levy - mc).abs() / mc;
  assert!(rel < 0.03, "levy={levy}, mc={mc}, rel={rel}");
}

/// MC geometric vs analytical geometric basket (should match closely).
#[cfg(feature = "openblas")]
#[test]
fn mc_geometric_matches_closed_form() {
  let (s, w, sig, q, rho) = iid_basket(3, 0.25, 0.5);
  let cf = GeometricBasketPricer::new(w.clone(), sig.clone(), rho.clone()).price_call(
    s.view(),
    100.0,
    0.05,
    q.view(),
    1.0,
  );
  let mc = McBasketPricer::new(w, BasketAverageType::Geometric, sig, rho, 200_000).price_call(
    s.view(),
    100.0,
    0.05,
    q.view(),
    1.0,
  );
  let rel = (cf - mc).abs() / cf;
  assert!(rel < 0.02, "cf={cf}, mc={mc}");
}

/// The two closed-form basket constructors now reject a shape mismatch, a
/// negative volatility and an out-of-range correlation entry; the Monte
/// Carlo one rejects only the volatilities and the path count.
///
/// The shape check is the one with the sharpest evidence, and it comes
/// from the Levy pricer: it has no dimension assertion anywhere, and its
/// moment loops run over the *query*'s asset count while indexing the
/// model's vectors, so a surplus model entry is silently discarded and a
/// short one silently truncates the basket.
mod construction_validation {
  use super::*;

  /// `10.894912090686852` — bit-identical to the healthy two-asset price,
  /// with the third volatility silently ignored.
  #[test]
  #[should_panic(
    expected = "ArithmeticBasketLevyPricer::new: weights, sigma and rho must agree on the asset count"
  )]
  fn levy_rejects_a_surplus_volatility() {
    let _ = ArithmeticBasketLevyPricer::new(
      array![0.5, 0.5],
      array![0.20, 0.30, 0.10],
      array![[1.0, 0.4], [0.4, 1.0]],
    );
  }

  /// A 3x3 correlation against a 2-asset model returned `9.783632`, using
  /// the top-left block and nothing else.
  #[test]
  #[should_panic(
    expected = "GeometricBasketPricer::new: weights, sigma and rho must agree on the asset count"
  )]
  fn geometric_rejects_a_correlation_of_the_wrong_size() {
    let _ = GeometricBasketPricer::new(array![0.5, 0.5], array![0.20, 0.30], Array2::<f64>::eye(3));
  }

  /// `8.487146` against `10.894912`.
  #[test]
  #[should_panic(
    expected = "ArithmeticBasketLevyPricer::new: sigma[0] must be a non-negative volatility (got -0.2)"
  )]
  fn levy_rejects_a_negative_volatility() {
    let _ = ArithmeticBasketLevyPricer::new(
      array![0.5, 0.5],
      array![-0.20, 0.30],
      array![[1.0, 0.4], [0.4, 1.0]],
    );
  }

  /// `6.946344` against `10.224832`.
  #[test]
  #[should_panic(
    expected = "GeometricBasketPricer::new: sigma[0] must be a non-negative volatility (got -0.2)"
  )]
  fn geometric_rejects_a_negative_volatility() {
    let _ = GeometricBasketPricer::new(
      array![0.5, 0.5],
      array![-0.20, 0.30],
      array![[1.0, 0.4], [0.4, 1.0]],
    );
  }

  /// The `sigma_g_sq.max(0.0)` floor swallows a negative basket variance
  /// and the geometric call comes back **`0.0`** — the `f64::max` trap
  /// again, this time on a model parameter rather than a price.
  #[test]
  #[should_panic(expected = "GeometricBasketPricer::new: rho[0][1] must be in [-1, 1] (got -5)")]
  fn geometric_rejects_a_correlation_below_minus_one() {
    let _ = GeometricBasketPricer::new(
      array![0.5, 0.5],
      array![0.20, 0.30],
      array![[1.0, -5.0], [-5.0, 1.0]],
    );
  }

  /// `4.877058` — the basket's zero-volatility intrinsic, the same number
  /// a `NaN` `sigma` produced before the Levy variance floor was split.
  #[test]
  #[should_panic(
    expected = "ArithmeticBasketLevyPricer::new: rho[0][1] must be in [-1, 1] (got -5)"
  )]
  fn levy_rejects_a_correlation_below_minus_one() {
    let _ = ArithmeticBasketLevyPricer::new(
      array![0.5, 0.5],
      array![0.20, 0.30],
      array![[1.0, -5.0], [-5.0, 1.0]],
    );
  }

  /// The range test covers the diagonal, so a matrix carrying variances
  /// instead of correlations is caught by the same check: `16.837128`
  /// against `10.224832`.
  #[test]
  #[should_panic(expected = "GeometricBasketPricer::new: rho[0][0] must be in [-1, 1] (got 3)")]
  fn geometric_rejects_a_correlation_diagonal_that_is_not_one() {
    let _ = GeometricBasketPricer::new(
      array![0.5, 0.5],
      array![0.20, 0.30],
      array![[3.0, 0.4], [0.4, 3.0]],
    );
  }

  #[cfg(feature = "openblas")]
  #[test]
  #[should_panic(
    expected = "McBasketPricer::new: sigma[1] must be a non-negative volatility (got -0.3)"
  )]
  fn mc_basket_rejects_a_negative_volatility() {
    let _ = McBasketPricer::new(
      array![0.5, 0.5],
      BasketAverageType::Arithmetic,
      array![0.25, -0.30],
      array![[1.0, 0.4], [0.4, 1.0]],
      1_000,
    );
  }

  #[cfg(feature = "openblas")]
  #[test]
  #[should_panic(expected = "McBasketPricer::new: n_paths must be at least 1 (got 0)")]
  fn mc_basket_rejects_a_zero_path_count() {
    let _ = McBasketPricer::new(
      array![0.5, 0.5],
      BasketAverageType::Arithmetic,
      array![0.25, 0.30],
      array![[1.0, 0.4], [0.4, 1.0]],
      0,
    );
  }

  /// The deliberate omissions, pinned so they cannot drift.
  ///
  /// The **weight sum** is free on both closed forms: a long/short basket
  /// is a real product, and `w = [-1, 2]` prices at `29.444371` /
  /// `22.810134` rather than at nonsense.
  ///
  /// **Symmetry** of `rho` is free, because an exact-equality test would
  /// reject an estimator's round-off-symmetric matrix. The residual is
  /// asserted rather than described: the geometric basket symmetrises an
  /// asymmetric `rho` *bit-identically*, and the Levy basket does not.
  ///
  /// `McBasketPricer` leaves `rho` and the shapes to `try_price`, which
  /// `mc_basket_try_price_reports_a_query_dimension_mismatch` needs.
  #[test]
  fn the_deliberate_omissions_stay_constructible() {
    let s = array![100.0, 100.0];
    let q = array![0.0, 0.0];
    let rho = array![[1.0, 0.4], [0.4, 1.0]];
    let long_short = GeometricBasketPricer::new(array![-1.0, 2.0], array![0.20, 0.30], rho.clone());
    assert!(long_short.price_call(s.view(), 100.0, 0.05, q.view(), 1.0) > 0.0);

    let asym = array![[1.0, 0.4], [0.9, 1.0]];
    let symm = array![[1.0, 0.65], [0.65, 1.0]];
    let w = array![0.5, 0.5];
    let sig = array![0.20, 0.30];
    let geo_asym = GeometricBasketPricer::new(w.clone(), sig.clone(), asym.clone()).price_call(
      s.view(),
      100.0,
      0.05,
      q.view(),
      1.0,
    );
    let geo_symm = GeometricBasketPricer::new(w.clone(), sig.clone(), symm.clone()).price_call(
      s.view(),
      100.0,
      0.05,
      q.view(),
      1.0,
    );
    assert_eq!(
      geo_asym, geo_symm,
      "the geometric basket silently symmetrises rho"
    );
    let levy_asym = ArithmeticBasketLevyPricer::new(w.clone(), sig.clone(), asym).price_call(
      s.view(),
      100.0,
      0.05,
      q.view(),
      1.0,
    );
    let levy_symm = ArithmeticBasketLevyPricer::new(w, sig, symm).price_call(
      s.view(),
      100.0,
      0.05,
      q.view(),
      1.0,
    );
    assert_ne!(
      levy_asym, levy_symm,
      "the Levy basket exponentiates each entry, so it does not"
    );
  }

  /// The accessor guards stay, because the fields are `pub` and so the
  /// constructor is a front door and not a wall. A query whose asset count
  /// disagrees with an internally consistent model is still the accessor's
  /// to catch, and its message is not a substring of the constructor's.
  #[test]
  #[should_panic(expected = "assertion `left == right` failed")]
  fn the_query_length_check_stays_at_the_accessor() {
    let model = GeometricBasketPricer::new(
      array![0.5, 0.5],
      array![0.20, 0.30],
      array![[1.0, 0.4], [0.4, 1.0]],
    );
    let s = array![100.0, 100.0, 100.0];
    let q = array![0.0, 0.0, 0.0];
    let _ = model.price_call(s.view(), 100.0, 0.05, q.view(), 1.0);
  }
}

/// A `NaN` *model* parameter used to price at the basket's **zero-volatility
/// intrinsic**, which is the sharpest form this trap takes anywhere in the
/// crate: nothing in the query is wrong and the answer is a plausible ATM
/// call.
///
/// `(m2 / m1²).ln().max(1e-14)` turned the `NaN` log-ratio into `1e-14`, so
/// `sigma_eff ~ 1e-7` and the Levy approximation collapsed onto the
/// deterministic-basket value. The measurement that identifies it: the
/// poisoned call and a genuinely zero-volatility basket at the same query
/// return the *same* number, `4.877057549928611`, against `10.894912…` for
/// the healthy model.
///
/// Both poisoned parameters are written straight to the `pub` fields
/// rather than passed to the constructor, so this stays a statement about
/// the estimator rather than about what `new` accepts.
///
/// The sibling [`GeometricBasketPricer`] is clean under the same probe —
/// it has no such floor — and is checked here so the fix is not read as
/// something the whole file needed.
#[test]
fn arithmetic_basket_levy_does_not_launder_a_nan_into_the_zero_vol_intrinsic() {
  let (s, w, sig, q, rho) = iid_basket(2, 0.20, 0.4);
  let healthy = ArithmeticBasketLevyPricer::new(w.clone(), sig.clone(), rho.clone());
  let live = healthy.price_call(s.view(), 100.0, 0.05, q.view(), 1.0);

  // The number the poisoned model used to impersonate.
  let frozen = ArithmeticBasketLevyPricer::new(w.clone(), Array1::zeros(2), rho.clone());
  let intrinsic = frozen.price_call(s.view(), 100.0, 0.05, q.view(), 1.0);
  assert!(
    intrinsic > 0.0 && intrinsic < live,
    "the zero-vol intrinsic {intrinsic} must be a plausible price below the live one {live}"
  );

  let mut poisoned = healthy.clone();
  poisoned.sigma[0] = f64::NAN;
  let got = poisoned.price_call(s.view(), 100.0, 0.05, q.view(), 1.0);
  assert!(got.is_nan(), "a NaN model sigma must not price: got {got}");
  let got = poisoned.price_put(s.view(), 100.0, 0.05, q.view(), 1.0);
  assert!(
    got.is_nan(),
    "a NaN model sigma must not price a put: {got}"
  );

  let mut poisoned = healthy.clone();
  poisoned.rho[[0, 1]] = f64::NAN;
  let got = poisoned.price_call(s.view(), 100.0, 0.05, q.view(), 1.0);
  assert!(got.is_nan(), "a NaN model rho must not price: got {got}");

  // The `1e-14` floor is untouched where it belongs: a basket with no
  // volatility at all still prices, rather than dividing by zero.
  assert!(intrinsic.is_finite());

  // The query-side `NaN`s already propagated and must keep doing so.
  let mut nan_spot = s.clone();
  nan_spot[0] = f64::NAN;
  assert!(
    healthy
      .price_call(nan_spot.view(), 100.0, 0.05, q.view(), 1.0)
      .is_nan()
  );
  assert!(
    healthy
      .price_call(s.view(), 100.0, 0.05, q.view(), f64::NAN)
      .is_nan()
  );

  // Clean under the same probe, and deliberately untouched.
  let mut geo = GeometricBasketPricer::new(w, sig, rho);
  geo.sigma[0] = f64::NAN;
  assert!(
    geo
      .price_call(s.view(), 100.0, 0.05, q.view(), 1.0)
      .is_nan(),
    "the geometric basket has no such floor and already propagated"
  );
}

/// Arithmetic basket put-call parity: $C - P = e^{-rT}(F - K)$ where
/// $F = E[B]$.
#[test]
fn arithmetic_basket_parity() {
  let (s, w, sig, q, rho) = iid_basket(3, 0.25, 0.3);
  let r = 0.04;
  let tau = 1.0;
  let k = 95.0;
  let model = ArithmeticBasketLevyPricer::new(w.clone(), sig.clone(), rho.clone());
  let c = model.price_call(s.view(), k, r, q.view(), tau);
  let p = model.price_put(s.view(), k, r, q.view(), tau);
  let f = first_moment(s.view(), w.view(), q.view(), r, tau);
  let lhs = c - p;
  let rhs = (-r * tau).exp() * (f - k);
  assert!((lhs - rhs).abs() < 0.01, "lhs={lhs}, rhs={rhs}");
}

/// One Monte Carlo model instance prices a whole strike grid, both legs.
/// The strikes are far enough apart that the ordering survives the sampling
/// error of independent simulations.
#[cfg(feature = "openblas")]
#[test]
fn mc_basket_one_model_prices_a_strike_grid() {
  let (s, w, sig, q, rho) = iid_basket(3, 0.25, 0.4);
  let model = McBasketPricer::new(w, BasketAverageType::Arithmetic, sig, rho, 50_000);
  let calls = [80.0, 100.0, 130.0].map(|k| model.price_call(s.view(), k, 0.05, q.view(), 1.0));
  let puts = [80.0, 100.0, 130.0].map(|k| model.price_put(s.view(), k, 0.05, q.view(), 1.0));
  assert!(
    calls[0] > calls[1] && calls[1] > calls[2],
    "basket calls must decay in the strike: {calls:?}"
  );
  assert!(
    puts[0] < puts[1] && puts[1] < puts[2],
    "basket puts must rise in the strike: {puts:?}"
  );
}

/// The model and the weights fix how many assets there are; a query that
/// disagrees is reported by `try_price` as an `Err`, not a panic. Pinned
/// because that is the reason the check did not move to the constructor.
#[cfg(feature = "openblas")]
#[test]
fn mc_basket_try_price_reports_a_query_dimension_mismatch() {
  let (_, w, sig, _, rho) = iid_basket(3, 0.25, 0.4);
  let model = McBasketPricer::new(w, BasketAverageType::Arithmetic, sig, rho, 1_000);
  let s = array![100.0, 100.0];
  let q = array![0.0, 0.0];
  let err = model
    .try_price(s.view(), 100.0, 0.05, q.view(), 1.0, OptionType::Call)
    .expect_err("a two-asset query against a three-asset model is not priceable");
  assert!(
    err.to_string().contains("does not match n_assets=2"),
    "{err}"
  );
}
