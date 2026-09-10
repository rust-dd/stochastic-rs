use super::*;
use basin::CostFunction;
use basin::Gradient;

fn bounded_problem() -> objective::SabrSmileProblem {
  objective::SabrSmileProblem {
    s: 1.0,
    r_d: 0.02,
    r_f: 0.01,
    tau: 0.5,
    beta: 1.0,
    sigma_atm: 0.2,
    sigma_rr: 0.01,
    sigma_bf: 0.002,
    bounds_lo: vec![0.5, 0.5, 0.5, 0.5, 0.01, -0.99],
    bounds_hi: vec![2.0, 2.0, 2.0, 2.0, 10.0, 0.99],
  }
}

#[test]
fn bounded_sabr_objective_clamps_strikes_and_model_parameters() {
  let problem = bounded_problem();
  let outside = vec![3.0, -0.1, 3.0, -0.1, 0.6, 1.2];
  let boundary = vec![2.0, 0.5, 2.0, 0.5, 0.6, 0.99];
  assert_eq!(
    problem.cost(&outside).unwrap(),
    problem.cost(&boundary).unwrap()
  );
  assert_eq!(
    problem.gradient(&outside).unwrap(),
    problem.gradient(&boundary).unwrap()
  );
}

#[test]
fn bounded_sabr_gradient_retains_the_derivative_at_the_upper_bound() {
  let problem = bounded_problem();
  let x = vec![2.0, 0.9, 1.1, 0.9, 0.6, 0.5];
  let mut inside = x.clone();
  inside[0] -= 1e-6;
  let expected = (problem.cost(&x).unwrap() - problem.cost(&inside).unwrap()) / (x[0] - inside[0]);
  let actual = problem.gradient(&x).unwrap()[0];
  assert!(expected.abs() > 1e-6);
  assert!((actual - expected).abs() < 1e-4 * expected.abs());
}

#[test]
fn bounded_sabr_low_spot_calibration_returns_consistent_results() {
  let quotes = SabrSmileQuotes {
    tau: 0.5,
    sigma_atm: 0.2,
    sigma_rr: 0.01,
    sigma_bf: 0.002,
  };
  for iterations in [0, 2] {
    let calibrator = SabrSmileCalibrator::new(0.05, 0.02, 0.01, 1.0, quotes)
      .with_strike_bounds(0.02, 0.08)
      .with_basin_hopping_iters(iterations, iterations);
    let result = calibrator.calibrate();
    let x = vec![
      result.k_rr_call,
      result.k_rr_put,
      result.k_bf_call,
      result.k_bf_put,
      result.params.nu,
      result.params.rho,
    ];
    let mut problem = bounded_problem();
    problem.s = calibrator.s;
    problem.bounds_lo[..4].fill(calibrator.strike_lo);
    problem.bounds_hi[..4].fill(calibrator.strike_hi);
    for i in 0..x.len() {
      assert!((problem.bounds_lo[i]..=problem.bounds_hi[i]).contains(&x[i]));
    }
    assert!(result.objective.is_finite());
    assert_eq!(result.objective, problem.cost(&x).unwrap());
    assert_eq!(result.success, result.objective < calibrator.success_tol);
  }
}

#[test]
fn test_sabr_smile_calibrate() {
  let r_usd = 0.022_f64;
  let r_brl = 0.065_f64;
  let s = 3.724_f64;
  let beta = 1.0;

  let cases: [(&str, SabrSmileQuotes); 8] = [
    (
      "ON",
      SabrSmileQuotes {
        tau: 1.0 / 365.0,
        sigma_atm: 20.98 / 100.0,
        sigma_rr: 1.2 / 100.0,
        sigma_bf: 0.15 / 100.0,
      },
    ),
    (
      "1W",
      SabrSmileQuotes {
        tau: 7.0 / 365.0,
        sigma_atm: 13.91 / 100.0,
        sigma_rr: 1.3 / 100.0,
        sigma_bf: 0.20 / 100.0,
      },
    ),
    (
      "2W",
      SabrSmileQuotes {
        tau: 14.0 / 365.0,
        sigma_atm: 13.75 / 100.0,
        sigma_rr: 1.4 / 100.0,
        sigma_bf: 0.20 / 100.0,
      },
    ),
    (
      "1M",
      SabrSmileQuotes {
        tau: 30.0 / 365.0,
        sigma_atm: 14.24 / 100.0,
        sigma_rr: 1.5 / 100.0,
        sigma_bf: 0.22 / 100.0,
      },
    ),
    (
      "2M",
      SabrSmileQuotes {
        tau: 60.0 / 365.0,
        sigma_atm: 13.84 / 100.0,
        sigma_rr: 1.75 / 100.0,
        sigma_bf: 0.27 / 100.0,
      },
    ),
    (
      "3M",
      SabrSmileQuotes {
        tau: 90.0 / 365.0,
        sigma_atm: 13.82 / 100.0,
        sigma_rr: 2.0 / 100.0,
        sigma_bf: 0.32 / 100.0,
      },
    ),
    (
      "6M",
      SabrSmileQuotes {
        tau: 180.0 / 365.0,
        sigma_atm: 13.82 / 100.0,
        sigma_rr: 2.4 / 100.0,
        sigma_bf: 0.43 / 100.0,
      },
    ),
    (
      "1Y",
      SabrSmileQuotes {
        tau: 1.0,
        sigma_atm: 13.94 / 100.0,
        sigma_rr: 2.9 / 100.0,
        sigma_bf: 0.55 / 100.0,
      },
    ),
  ];

  let results = SabrSmileCalibrator::calibrate_and_plot_many(s, r_brl, r_usd, beta, &cases);

  for (i, ((label, q), res)) in cases.iter().zip(results.iter()).enumerate() {
    println!("\nTenor {} (T={:.4}):", label, q.tau);
    println!(
      "  K_ATM={:.6}, alpha={:.6}, beta={:.2}, nu={:.6}, rho={:.6}",
      res.k_atm, res.params.alpha, res.params.beta, res.params.nu, res.params.rho
    );
    println!("  Objective: {:.6e}", res.objective);
    assert!(res.success);
    assert!(res.objective < 1e-3, "Objective too large for tenor {}", i);
  }
}
