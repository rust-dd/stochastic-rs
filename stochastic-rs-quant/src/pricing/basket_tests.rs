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
  let ari = ArithmeticBasketLevyPricer {
    s,
    weights: w,
    sigma: sig,
    q,
    rho,
    k: 100.0,
    r: 0.04,
    tau: 1.0,
    option_type: OptionType::Call,
  }
  .price();
  assert!(geo < ari, "geo={geo} should be < ari={ari}");
}

/// Levy and MC should agree within ~3% for a 4-asset arithmetic basket.
#[cfg(feature = "openblas")]
#[test]
fn levy_vs_mc_arithmetic() {
  let (s, w, sig, q, rho) = iid_basket(4, 0.25, 0.4);
  let levy = ArithmeticBasketLevyPricer {
    s: s.clone(),
    weights: w.clone(),
    sigma: sig.clone(),
    q: q.clone(),
    rho: rho.clone(),
    k: 100.0,
    r: 0.05,
    tau: 1.0,
    option_type: OptionType::Call,
  }
  .price();
  let mc = McBasketPricer {
    s,
    weights: w,
    sigma: sig,
    q,
    rho,
    k: 100.0,
    r: 0.05,
    tau: 1.0,
    option_type: OptionType::Call,
    avg_type: BasketAverageType::Arithmetic,
    n_paths: 100_000,
  }
  .price();
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
  let mc = McBasketPricer {
    s,
    weights: w,
    sigma: sig,
    q,
    rho,
    k: 100.0,
    r: 0.05,
    tau: 1.0,
    option_type: OptionType::Call,
    avg_type: BasketAverageType::Geometric,
    n_paths: 200_000,
  }
  .price();
  let rel = (cf - mc).abs() / cf;
  assert!(rel < 0.02, "cf={cf}, mc={mc}");
}

/// Arithmetic basket put-call parity: $C - P = e^{-rT}(F - K)$ where
/// $F = E[B]$.
#[test]
fn arithmetic_basket_parity() {
  let (s, w, sig, q, rho) = iid_basket(3, 0.25, 0.3);
  let r = 0.04;
  let tau = 1.0;
  let k = 95.0;
  let c = ArithmeticBasketLevyPricer {
    s: s.clone(),
    weights: w.clone(),
    sigma: sig.clone(),
    q: q.clone(),
    rho: rho.clone(),
    k,
    r,
    tau,
    option_type: OptionType::Call,
  }
  .price();
  let p = ArithmeticBasketLevyPricer {
    s: s.clone(),
    weights: w.clone(),
    sigma: sig.clone(),
    q: q.clone(),
    rho: rho.clone(),
    k,
    r,
    tau,
    option_type: OptionType::Put,
  }
  .price();
  let f = first_moment(s.view(), w.view(), q.view(), r, tau);
  let lhs = c - p;
  let rhs = (-r * tau).exp() * (f - k);
  assert!((lhs - rhs).abs() < 0.01, "lhs={lhs}, rhs={rhs}");
}
