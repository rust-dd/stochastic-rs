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

/// Geometric basket on a single asset must equal a vanilla BSM call.
#[test]
fn geometric_basket_n1_matches_bsm() {
  let p = GeometricBasketPricer {
    s: array![100.0],
    weights: array![1.0],
    sigma: array![0.2],
    q: array![0.0],
    rho: array![[1.0]],
    k: 100.0,
    r: 0.05,
    tau: 1.0,
    option_type: OptionType::Call,
  };
  let price = p.price();
  let bsm = 10.4506;
  assert!((price - bsm).abs() < 0.005, "geo n=1: {price}");
}

/// Geometric basket with perfectly correlated identical assets equals
/// a single-asset vanilla.
#[test]
fn geometric_basket_perfect_corr_equals_single() {
  let (s, w, sig, q, rho) = iid_basket(5, 0.20, 1.0);
  let p = GeometricBasketPricer {
    s,
    weights: w,
    sigma: sig,
    q,
    rho,
    k: 100.0,
    r: 0.05,
    tau: 1.0,
    option_type: OptionType::Call,
  };
  let price = p.price();
  let bsm = 10.4506;
  assert!((price - bsm).abs() < 0.01, "geo perf-corr: {price}");
}

/// Geometric basket should be cheaper than arithmetic (Jensen).
#[test]
fn geometric_below_arithmetic() {
  let (s, w, sig, q, rho) = iid_basket(4, 0.30, 0.5);
  let geo = GeometricBasketPricer {
    s: s.clone(),
    weights: w.clone(),
    sigma: sig.clone(),
    q: q.clone(),
    rho: rho.clone(),
    k: 100.0,
    r: 0.04,
    tau: 1.0,
    option_type: OptionType::Call,
  }
  .price();
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
  let cf = GeometricBasketPricer {
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
