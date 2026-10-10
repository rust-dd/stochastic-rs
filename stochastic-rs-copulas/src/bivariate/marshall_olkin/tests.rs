use ndarray::array;

use super::*;

fn approx(a: f64, b: f64, tol: f64) -> bool {
  (a - b).abs() <= tol
}

#[test]
fn mo_cdf_marginal_recovers_input() {
  let c = MarshallOlkin::with_alpha_beta(0.5, 0.3);
  let x = array![[0.4_f64, 1.0], [1.0, 0.7]];
  let cdf = c.cdf(&x).unwrap();
  assert!(approx(cdf[0], 0.4, 1e-12));
  assert!(approx(cdf[1], 0.7, 1e-12));
}

#[test]
fn mo_alpha_eq_one_beta_eq_one_is_comonotone() {
  let c = MarshallOlkin::with_alpha_beta(1.0, 1.0);
  let x = array![[0.3_f64, 0.7], [0.6, 0.2], [0.5, 0.5]];
  let cdf = c.cdf(&x).unwrap();
  for i in 0..x.nrows() {
    let expected = x[[i, 0]].min(x[[i, 1]]);
    assert!(approx(cdf[i], expected, 1e-12), "row {i}");
  }
}

#[test]
fn mo_alpha_zero_or_beta_zero_is_independence() {
  // α → 0 with β fixed: C(u,v) = u^{1-0} v = u v. Use α just above 0.
  let c = MarshallOlkin::with_alpha_beta(1e-12, 0.5);
  let x = array![[0.4_f64, 0.6]];
  let cdf = c.cdf(&x).unwrap();
  assert!(approx(cdf[0], 0.24, 1e-6), "α→0: got {}", cdf[0]);
}

#[test]
fn mo_compute_theta_via_symmetric_inversion() {
  // Symmetric MO: τ = θ/(2-θ). Pick θ = 0.5 ⇒ τ = 1/3; invert to recover.
  let mut c = MarshallOlkin::new();
  c.set_tau(1.0 / 3.0);
  let theta = c.compute_theta();
  assert!(approx(theta, 0.5, 1e-12), "expected θ=0.5, got {theta}");
}

#[test]
fn mo_singular_curve_total_mass_matches_paper() {
  // Singular component carries mass αβ/(α+β-αβ). Verify against
  // Monte-Carlo on a fine grid: count fraction of unit-square sectors
  // dominated by the absolutely continuous density vs total.
  let alpha = 0.6_f64;
  let beta = 0.4_f64;
  let mass_singular_paper = alpha * beta / (alpha + beta - alpha * beta);

  // Integrate the absolutely continuous density on a 200×200 grid.
  let c = MarshallOlkin::with_alpha_beta(alpha, beta);
  let n = 200usize;
  let h = 1.0 / n as f64;
  let mut points = Array2::<f64>::zeros((n * n, 2));
  for i in 0..n {
    for j in 0..n {
      let row = i * n + j;
      points[[row, 0]] = (i as f64 + 0.5) * h;
      points[[row, 1]] = (j as f64 + 0.5) * h;
    }
  }
  let pdf_vals = c.pdf(&points).unwrap();
  let mass_abs: f64 = pdf_vals.iter().sum::<f64>() * h * h;
  let mass_singular_grid = 1.0 - mass_abs;

  assert!(
    (mass_singular_grid - mass_singular_paper).abs() < 0.02,
    "singular mass grid={mass_singular_grid:.4}, paper={mass_singular_paper:.4}"
  );
}

/// `∂_v C` from mpmath's derivative of `C` in `v`, 17 digits, in both sectors.
#[test]
fn mo_partial_derivative_is_the_v_derivative_in_each_sector() {
  let cases = [
    (0.5, 0.5, 0.9, 0.5, 0.9486832980505138),
    (0.5, 0.5, 0.3, 0.8, 0.16770509831248423),
    (0.5, 0.5, 0.6, 0.4, 0.7745966692414834),
    (0.3, 0.6, 0.9, 0.5, 0.928901697685371),
    (0.3, 0.6, 0.3, 0.8, 0.1371915155781979),
    (0.3, 0.6, 0.6, 0.4, 0.6993681904144294),
    (0.8, 0.2, 0.9, 0.5, 0.9791483623609768),
    (0.8, 0.2, 0.3, 0.8, 0.2509534926219056),
    (0.8, 0.2, 0.6, 0.4, 0.576539728311087),
  ];
  for (alpha, beta, u, v, want) in cases {
    let c = MarshallOlkin::with_alpha_beta(alpha, beta);
    let h = c.partial_derivative(&array![[u, v]]).unwrap()[0];
    assert!(
      (h - want).abs() < 1e-15,
      "α={alpha} β={beta} (u,v)=({u},{v}): {h} vs {want}"
    );
  }
}

/// The generalised inverse at `v = 0.37`: both continuous pieces and the atom, with the `α = 1`, `β = 1` and
/// `α = β = 1` branches.
#[test]
fn mo_percent_point_matches_the_reference_table() {
  let v = 0.37_f64;
  let cases = [
    (0.5, 0.5, 0.25 * v.sqrt(), 0.185),
    (0.5, 0.5, 0.5 * v.sqrt(), 0.37),
    (0.5, 0.5, 0.45620719, 0.37),
    (0.5, 0.5, 0.608276253, 0.37),
    (0.5, 0.5, 0.5 + 0.5 * v.sqrt(), 0.6466381265149109),
    (0.3, 0.6, 0.2 * v.powf(1.4), 0.06845),
    (0.3, 0.6, 0.1740125, 0.1369),
    (0.3, 0.6, 0.5 + 0.5 * v.powf(1.4), 0.5101518223511724),
    (1.0, 0.4, 0.3, 0.33593147277382094),
    (1.0, 0.4, 0.8, 0.6718629455476419),
    (0.7, 1.0, 0.32652287, 0.2416269211589283),
    (0.7, 1.0, 0.5 + 0.5 * v.powf(3.0 / 7.0), 0.5298865677782774),
    (1.0, 1.0, 0.5, 0.37),
  ];
  for (alpha, beta, y, want) in cases {
    let c = MarshallOlkin::with_alpha_beta(alpha, beta);
    let u = c.percent_point(&array![y], &array![v]).unwrap()[0];
    assert!(
      (u - want).abs() < 1e-14,
      "α={alpha} β={beta} y={y}: {u} vs {want}"
    );
  }
}

/// Off the atom the inverse round-trips through `h`; on the atom `h(u−) ≤ y ≤ h(u)`.
#[test]
fn mo_percent_point_inverts_the_h_function() {
  for (alpha, beta) in [(0.5, 0.5), (0.3, 0.6), (0.8, 0.2), (1.0, 0.4), (0.7, 1.0)] {
    let c = MarshallOlkin::with_alpha_beta(alpha, beta);
    for v in [0.05_f64, 0.37, 0.9] {
      let w = v.powf(beta * (1.0 - alpha) / alpha);
      for y in [0.02, 0.2, 0.5, 0.8, 0.98] {
        let u = c.percent_point(&array![y], &array![v]).unwrap()[0];
        let h_at = c.partial_derivative(&array![[u, v]]).unwrap()[0];
        if y < (1.0 - beta) * w || y > w {
          assert!(
            (h_at - y).abs() < 1e-12,
            "α={alpha} β={beta} v={v} y={y}: h({u})={h_at}"
          );
        } else {
          let h_left = c.partial_derivative(&array![[u - 1e-12, v]]).unwrap()[0];
          assert!(
            h_left <= y + 1e-9 && y <= h_at + 1e-9,
            "atom: {h_left} ≤ {y} ≤ {h_at}"
          );
        }
      }
    }
  }
}

/// A NaN or out-of-range `y` or `v` is NaN, which the raw formulas would map to a plausible number or ±∞.
#[test]
fn mo_a_query_outside_the_unit_interval_is_nan() {
  for (alpha, beta) in [(0.5, 0.5), (0.3, 0.6), (1.0, 0.4), (0.7, 1.0)] {
    let c = MarshallOlkin::with_alpha_beta(alpha, beta);
    for bad in [f64::NAN, 1.5, -0.1] {
      let u = c
        .percent_point(&array![bad, 0.5], &array![0.5, bad])
        .unwrap();
      assert!(
        u.iter().all(|x| x.is_nan()),
        "α={alpha} β={beta} {bad}: {u}"
      );
      let h = c.partial_derivative(&array![[0.5, bad]]).unwrap()[0];
      assert!(h.is_nan(), "α={alpha} β={beta} v={bad}: h={h}");
    }
    let u = c
      .percent_point(&array![0.0, 1.0, 0.5, 0.5], &array![0.5, 0.5, 0.0, 1.0])
      .unwrap();
    let h = c
      .partial_derivative(&array![[0.5, 0.0], [0.5, 1.0]])
      .unwrap();
    assert!(
      u.iter().chain(&h).all(|x| x.is_finite()),
      "α={alpha} β={beta}: edges {u} {h}"
    );
  }
}

/// τ, the singular mass and the cdf on a grid, against Nelsen's closed forms; no `Numerical` failure at any `n`.
#[test]
fn mo_sampler_reproduces_tau_the_singular_mass_and_the_cdf() {
  for n in [1_usize, 5, 20, 200] {
    assert!(
      MarshallOlkin::with_alpha_beta(0.5, 0.5)
        .sample_with_seed(n, 7)
        .is_ok(),
      "n = {n}"
    );
  }
  let grid = [(0.2, 0.3), (0.5, 0.5), (0.7, 0.4), (0.3, 0.9), (0.85, 0.65)];
  for (alpha, beta) in [(0.5, 0.5), (0.3, 0.6), (0.6, 0.4), (1.0, 0.4), (0.7, 1.0)] {
    let c = MarshallOlkin::with_alpha_beta(alpha, beta);
    let n = 40_000_usize;
    let uv = c.sample_with_seed(n, 7).unwrap();
    let (u, v) = (uv.column(0).to_vec(), uv.column(1).to_vec());
    let (tau, ..) =
      kendalls::tau_b_with_comparator(&u, &v, |a: &f64, b: &f64| a.partial_cmp(b).unwrap())
        .unwrap();
    let want = alpha * beta / (alpha + beta - alpha * beta);
    assert!(
      (tau - want).abs() < 0.01,
      "α={alpha} β={beta}: τ {tau} vs {want}"
    );
    let atoms = u
      .iter()
      .zip(&v)
      .filter(|(a, b)| (a.powf(alpha) - b.powf(beta)).abs() < 1e-9)
      .count() as f64
      / n as f64;
    assert!(
      (atoms - want).abs() < 0.01,
      "α={alpha} β={beta}: atom share {atoms} vs {want}"
    );
    for (gu, gv) in grid {
      let empirical = u
        .iter()
        .zip(&v)
        .filter(|(a, b)| **a <= gu && **b <= gv)
        .count() as f64
        / n as f64;
      let exact = (gu.powf(1.0 - alpha) * gv).min(gu * gv.powf(1.0 - beta));
      assert!(
        (empirical - exact).abs() < 0.01,
        "α={alpha} β={beta} ({gu},{gv}): {empirical} vs {exact}"
      );
    }
  }
  let comonotone = MarshallOlkin::with_alpha_beta(1.0, 1.0)
    .sample_with_seed(500, 7)
    .unwrap();
  assert!(comonotone.rows().into_iter().all(|r| r[0] == r[1]));
}

#[test]
fn mo_tail_dependence_matches_min_alpha_beta() {
  let c = MarshallOlkin::with_alpha_beta(0.6, 0.4);
  let td = c.tail_dependence();
  assert!(approx(td.upper, 0.4, 1e-12), "got {}", td.upper);
  assert_eq!(td.lower, 0.0);
}

/// A raw `set_theta(5.0)` bypasses `with_alpha_beta`'s constructor
/// asserts, resolving to `(alpha, beta) = (5.0, 5.0)` — outside `(0, 1]`.
/// Must panic, not silently report `λ_U = 5.0`.
#[test]
#[should_panic(expected = "tail_dependence requires a valid theta")]
fn mo_tail_dependence_panics_on_invalid_theta() {
  let mut c = MarshallOlkin::new();
  c.set_theta(5.0);
  let _ = c.tail_dependence();
}

/// pdf/cdf/partial_derivative on an unfit `MarshallOlkin` (neither
/// `theta` nor `(alpha, beta)` set) must return `Err`, matching every
/// sibling copula's `check_fit`-gated contract — not panic via
/// `resolve_params().expect(..)`.
#[test]
fn marshall_olkin_unfit_errs_like_siblings() {
  let c = MarshallOlkin::new();
  let x = array![[0.4_f64, 0.6]];
  assert!(c.pdf(&x).is_err(), "pdf must Err, not panic, when unfit");
  assert!(c.cdf(&x).is_err(), "cdf must Err, not panic, when unfit");
  assert!(
    c.partial_derivative(&x).is_err(),
    "partial_derivative must Err, not panic, when unfit"
  );
}

/// `generator` has no override in this family, so this exercises
/// `BivariateExt::generator`'s trait-default body directly.
#[test]
fn marshall_olkin_generator_returns_err_not_archimedean() {
  let c = MarshallOlkin::with_alpha_beta(0.5, 0.3);
  let t = array![0.5_f64, 0.8];
  assert!(c.generator(&t).is_err());
}

#[test]
fn an_out_of_domain_parameter_neither_samples_nor_evaluates() {
  let x = array![[0.3_f64, 0.4]];
  for theta in [1.5, -0.5] {
    let c = MarshallOlkin {
      theta: Some(theta),
      ..MarshallOlkin::new()
    };
    assert!(
      matches!(
        c.sample_with_seed(16, 7),
        Err(CopulaError::InvalidParameter { name: "theta", .. })
      ),
      "theta = {theta}"
    );
    assert!(
      matches!(
        c.pdf(&x),
        Err(CopulaError::InvalidParameter { name: "theta", .. })
      ),
      "theta = {theta}"
    );
  }
  let c = MarshallOlkin {
    alpha: Some(1.5),
    beta: Some(0.5),
    ..MarshallOlkin::new()
  };
  assert!(matches!(
    c.cdf(&x),
    Err(CopulaError::InvalidParameter { name: "alpha", .. })
  ));
}
