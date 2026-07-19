use super::*;

fn digital_put(x: &[f64], k: [f64; 2]) -> f64 {
  if x.len() == 2 && x[0] >= 0.0 && x[0] <= k[0] && x[1] >= 0.0 && x[1] <= k[1] {
    1.0
  } else {
    0.0
  }
}

#[test]
fn sphere_area_known_values() {
  let a2 = sphere_area::<f64>(2);
  assert!((a2 - 2.0 * std::f64::consts::PI).abs() < 1e-10);
  let a3 = sphere_area::<f64>(3);
  assert!((a3 - 4.0 * std::f64::consts::PI).abs() < 1e-10);
}

#[test]
fn digital_put_g_preserves_logarithmic_corner_singularity() {
  let g = g_digital_put_2d([100.0_f64, 100.0], [100.0, 100.0]);
  assert!(g[0][0].is_finite());
  assert!(g[1][1].is_finite());
  assert!(g[0][1].is_infinite() && g[0][1].is_sign_negative());
  assert_eq!(g[0][1], g[1][0]);
}

#[test]
fn digital_put_g_has_equal_diagonal_inside_symmetric_rectangle() {
  let g = g_digital_put_2d([90.0_f64, 90.0], [100.0, 100.0]);
  assert!((g[0][0] - 0.5).abs() < 1e-14, "g11 = {}", g[0][0]);
  assert!((g[1][1] - 0.5).abs() < 1e-14, "g22 = {}", g[1][1]);
  assert_eq!(g[0][1], g[1][0]);
}

#[test]
fn digital_put_g_is_scale_invariant_away_from_singularities() {
  let y = [0.9_f64, 1.4];
  let k = [1.8_f64, 2.3];
  let expected = g_digital_put_2d(y, k);

  for scale in [1e-12, 1e-6, 1e6, 1e12] {
    let scaled = g_digital_put_2d([scale * y[0], scale * y[1]], [scale * k[0], scale * k[1]]);
    for i in 0..2 {
      for j in 0..2 {
        assert!(
          (scaled[i][j] - expected[i][j]).abs() < 2e-14,
          "scale={scale}, g[{i}][{j}]={}, expected={}",
          scaled[i][j],
          expected[i][j]
        );
      }
    }
  }
}

#[test]
fn digital_put_g_trace_equals_payoff_on_and_off_support() {
  let k = [100.0_f64, 100.0];
  let cases = [
    ([90.0, 90.0], 1.0),
    ([0.0, 50.0], 1.0),
    ([100.0, 50.0], 1.0),
    ([-1e-6, 50.0], 0.0),
    ([50.0, -1e-6], 0.0),
    ([100.0 + 1e-6, 50.0], 0.0),
    ([50.0, 100.0 + 1e-6], 0.0),
  ];

  for (y, expected) in cases {
    let g = g_digital_put_2d(y, k);
    let trace = g[0][0] + g[1][1];
    assert!((trace - expected).abs() < 1e-14, "y={y:?}, trace={trace}");
  }
}

#[test]
fn poisson_kernel_gradient_decays() {
  let g1 = grad_poisson_reg(&[1.0, 1.0], 0.01);
  let g2 = grad_poisson_reg(&[10.0, 10.0], 0.01);
  let n1 = g1.iter().map(|x| x * x).sum::<f64>().sqrt();
  let n2 = g2.iter().map(|x| x * x).sum::<f64>().sqrt();
  assert!(n1 > n2, "|∇Q(1)| = {n1} should exceed |∇Q(10)| = {n2}");
}

#[test]
fn poisson_kernel_gradient_matches_3d_newton_potential() {
  let x = [1.0_f64, 2.0, 2.0];
  let grad = grad_poisson_reg(&x, 0.0);
  let r = x.iter().map(|value| value * value).sum::<f64>().sqrt();
  let factor = 1.0 / (4.0 * std::f64::consts::PI * r.powi(3));

  for i in 0..3 {
    let expected = x[i] * factor;
    assert!(
      (grad[i] - expected).abs() < 1e-12,
      "grad[{i}] = {}, expected {expected}",
      grad[i]
    );
  }
}

#[test]
fn poisson_kernel_flux_is_one_in_dimensions_four_and_five() {
  for d in [4, 5] {
    let radius = 2.75_f64;
    let mut x = vec![0.0; d];
    x[0] = radius;
    let grad = grad_poisson_reg(&x, 0.0);
    let radial_derivative = grad[0];
    let flux = radial_derivative * sphere_area::<f64>(d) * radius.powi(d as i32 - 1);
    assert!((flux - 1.0).abs() < 1e-14, "d={d}, flux={flux}");
  }
}

#[test]
fn poisson_gradient_normalization_is_dimension_independent() {
  for d in [4, 5] {
    let x = (1..=d).map(|i| i as f64).collect::<Vec<_>>();
    let radius = x.iter().map(|value| value * value).sum::<f64>().sqrt();
    let grad = grad_poisson_reg(&x, 0.0);
    let factor = 1.0 / (sphere_area::<f64>(d) * radius.powi(d as i32));
    for i in 0..d {
      let expected = x[i] * factor;
      assert!((grad[i] - expected).abs() < 1e-15, "d={d}, i={i}");
    }
  }
}

#[test]
fn numerical_2d_and_nd_quadratures_agree() {
  let y = [0.7_f64, 0.4];
  let lo = [0.0_f64, 0.0];
  let hi = [1.5_f64, 1.5];
  let payoff = |x: &[f64]| digital_put(x, [1.0, 0.8]);
  let g_2d = g_kernel_numerical_2d(&y, &payoff, 0.05, &lo, &hi, 32);
  let g_nd = g_kernel_numerical_nd(&y, &payoff, 0.05, &lo, &hi, 32);

  for i in 0..2 {
    for j in 0..2 {
      assert!((g_2d[[i, j]] - g_nd[[i, j]]).abs() < 1e-14);
    }
  }
}

#[test]
fn numerical_quadrature_matches_closed_form_inside_and_outside_support() {
  let k = [1.0_f64, 1.0];
  let lo = [0.0_f64, 0.0];
  let hi = k;
  let payoff = |x: &[f64]| digital_put(x, k);

  for y in [[0.4, 0.6], [1.4, 0.6]] {
    let closed = g_digital_put_2d(y, k);
    let numerical = g_kernel_numerical_2d(&y, &payoff, 2e-4, &lo, &hi, 256);
    for i in 0..2 {
      for j in 0..2 {
        let error = (closed[i][j] - numerical[[i, j]]).abs();
        assert!(
          error < 0.001,
          "y={y:?}, g[{i}][{j}]: closed={}, numerical={}, error={error}",
          closed[i][j],
          numerical[[i, j]]
        );
      }
    }
  }
}

#[test]
fn regularized_kernel_trace_identity_holds() {
  let z = [2.0_f64, 3.0, 1.5];
  let h = 0.01;
  let d = z.len();
  let trace = (0..d).map(|i| kernel_k_ij_h(&z, h, i, i)).sum::<f64>();
  let radius_h = (z.iter().map(|x| x * x).sum::<f64>() + h).sqrt();
  let z_squared = z.iter().map(|x| x * x).sum::<f64>();
  let expected = (d as f64 / radius_h.powi(d as i32)
    - d as f64 * z_squared / radius_h.powi(d as i32 + 2))
    / sphere_area::<f64>(d);

  assert!(
    (trace - expected).abs() < 1e-10,
    "trace={trace}, expected={expected}"
  );
}

#[test]
fn regularized_kernel_is_symmetric_in_indices() {
  let z = [1.0_f64, 2.0, 3.0, 4.0];
  let h = 0.01;

  for i in 0..z.len() {
    for j in (i + 1)..z.len() {
      let kij = kernel_k_ij_h(&z, h, i, j);
      let kji = kernel_k_ij_h(&z, h, j, i);
      assert!(
        (kij - kji).abs() < 1e-14,
        "K[{i},{j}]={kij}, K[{j},{i}]={kji}"
      );
    }
  }
}
