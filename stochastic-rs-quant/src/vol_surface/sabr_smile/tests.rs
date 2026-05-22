use super::*;

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
