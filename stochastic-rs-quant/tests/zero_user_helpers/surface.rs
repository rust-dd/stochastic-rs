use ndarray::Array2;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_quant::pricing::dupire::Dupire;
use stochastic_rs_quant::vol_surface::SviRawParams;

fn grid() -> Vec<f64> {
  (0..201).map(|i| -1.5 + 3.0 * i as f64 / 200.0).collect()
}

/// The projection clamps `ρ` into the box and the answer is the butterfly check of the slice it
/// leaves.
#[test]
fn project_with_butterfly_check_answers_for_the_projected_slice() {
  let mut slice = SviRawParams::new(0.04, 0.1, 2.0, 0.0, 0.2);
  let free = slice.project_with_butterfly_check(&grid());
  assert_eq!(slice.rho, 0.9999);
  assert_eq!(free, slice.is_butterfly_arb_free(&grid()));
}

/// Gatheral & Jacquier (2014), Example 3.1 (Vogt's slice): admissible, so the projection leaves
/// it as it is, and not butterfly-free, so the answer is `false`.
#[test]
fn project_with_butterfly_check_is_false_for_an_admissible_slice_with_butterfly_arbitrage() {
  let mut slice = SviRawParams::new(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153);
  assert!(slice.is_admissible());
  assert!(!slice.project_with_butterfly_check(&grid()));
  assert_eq!((slice.a, slice.rho, slice.sigma), (-0.0410, 0.3060, 0.4153));
}

fn black_scholes_call(s: f64, k: f64, r: f64, sigma: f64, tau: f64) -> f64 {
  let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * tau) / (sigma * tau.sqrt());
  s * norm_cdf(d1) - k * (-r * tau).exp() * norm_cdf(d1 - sigma * tau.sqrt())
}

/// `local_vol_slice(j)` is row `j` of `local_vol_surface`, NaN cells included.
#[test]
fn local_vol_slice_is_a_row_of_the_local_vol_surface() {
  let ks = vec![80.0, 90.0, 100.0, 110.0, 120.0];
  let ts = vec![0.25, 0.5, 1.0];
  let calls = Array2::from_shape_fn((ts.len(), ks.len()), |(j, i)| {
    black_scholes_call(100.0, ks[i], 0.03, 0.2, ts[j])
  });
  let dupire = Dupire::builder(ks, ts.clone(), calls).r(0.03).build();
  let surface = dupire.local_vol_surface();
  for j in 0..ts.len() {
    let slice = dupire.local_vol_slice(j);
    let row = surface.row(j);
    assert_eq!(slice.len(), row.len());
    for (a, b) in slice.iter().zip(row) {
      assert!(
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
        "row {j}: {a} vs {b}"
      );
    }
  }
}
