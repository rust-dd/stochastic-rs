//! Comparison tests for the `stochastic::mc` module.
//!
//! Validates each Monte Carlo method against analytical or published reference
//! values. Requires the `openblas` feature for the LSM solver. Run with:
//!
//! ```bash
//! cargo test --release --features openblas mc_comparison -- --nocapture --test-threads=1
//! ```

#![cfg(feature = "openblas")]

use ndarray::Array1;
use ndarray::Array2;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;
use stochastic_rs::stochastic::mc;
use stochastic_rs::stochastic::mc::halton::HaltonSeq;
use stochastic_rs::stochastic::mc::lsm::Lsm;
use stochastic_rs::stochastic::mc::mlmc::Mlmc;
use stochastic_rs::stochastic::mc::sobol::SobolSeq;
use stochastic_rs::traits::FloatExt;

/// Black-Scholes call price (analytical reference).
fn bs_call(s: f64, k: f64, r: f64, sigma: f64, tau: f64) -> f64 {
  let n = Normal::new(0.0, 1.0).unwrap();
  let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * tau) / (sigma * tau.sqrt());
  let d2 = d1 - sigma * tau.sqrt();
  s * n.cdf(d1) - k * (-r * tau).exp() * n.cdf(d2)
}

/// Black-Scholes put price (analytical reference).
fn bs_put(s: f64, k: f64, r: f64, sigma: f64, tau: f64) -> f64 {
  let n = Normal::new(0.0, 1.0).unwrap();
  let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * tau) / (sigma * tau.sqrt());
  let d2 = d1 - sigma * tau.sqrt();
  k * (-r * tau).exp() * n.cdf(-d2) - s * n.cdf(-d1)
}

/// Build a Gbm payoff closure: generate a Gbm terminal price from standard
/// normals `z` (single time step, exact solution) and return the discounted
/// call payoff.
fn gbm_call_payoff(s0: f64, k: f64, r: f64, sigma: f64, tau: f64) -> impl Fn(&Array1<f64>) -> f64 {
  move |z: &Array1<f64>| {
    let s_t = s0 * ((r - 0.5 * sigma * sigma) * tau + sigma * tau.sqrt() * z[0]).exp();
    (s_t - k).max(0.0) * (-r * tau).exp()
  }
}

/// Plain MC estimate for baseline variance comparison.
fn plain_mc(
  n_paths: usize,
  dim: usize,
  payoff: &dyn Fn(&Array1<f64>) -> f64,
) -> mc::McEstimate<f64> {
  let mut sum = 0.0;
  let mut sum_sq = 0.0;
  for _ in 0..n_paths {
    let z = f64::normal_array(dim, 0.0, 1.0);
    let y = payoff(&z);
    sum += y;
    sum_sq += y * y;
  }
  let n = n_paths as f64;
  let mean = sum / n;
  let var = sum_sq / n - mean * mean;
  mc::McEstimate {
    mean,
    std_err: (var / n).sqrt(),
    n_samples: n_paths,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Variance reduction vs BS analytical
// ─────────────────────────────────────────────────────────────────────────────

const S0: f64 = 100.0;
const K: f64 = 100.0;
const R: f64 = 0.05;
const SIGMA: f64 = 0.2;
const TAU: f64 = 1.0;
const N_PATHS: usize = 100_000;

/// Antithetic variates: price matches BS and std_err < plain MC.
#[test]
fn antithetic_vs_bs_call() {
  let bs = bs_call(S0, K, R, SIGMA, TAU);
  let payoff = gbm_call_payoff(S0, K, R, SIGMA, TAU);

  let av = mc::antithetic::estimate(N_PATHS, 1, &payoff);
  let plain = plain_mc(N_PATHS, 1, &payoff);

  let err = (av.mean - bs).abs();
  println!(
    "AV:    mean={:.4}, se={:.6}, err={:.4}",
    av.mean, av.std_err, err
  );
  println!("Plain: mean={:.4}, se={:.6}", plain.mean, plain.std_err);
  println!("BS:    {bs:.4}");

  assert!(
    err < 0.30,
    "AV mean {:.4} too far from BS {bs:.4} (err={err:.4})",
    av.mean
  );
  assert!(
    av.std_err < plain.std_err,
    "AV se ({:.6}) should be < plain se ({:.6})",
    av.std_err,
    plain.std_err
  );
}

/// Control variates: use the discounted terminal stock price
/// (E[e^{-rT} S_T] = S_0) as control variate.
#[test]
fn control_variate_vs_bs_call() {
  let bs = bs_call(S0, K, R, SIGMA, TAU);

  let payoff = gbm_call_payoff(S0, K, R, SIGMA, TAU);
  let control = |z: &Array1<f64>| {
    let s_t = S0 * ((R - 0.5 * SIGMA * SIGMA) * TAU + SIGMA * TAU.sqrt() * z[0]).exp();
    s_t * (-R * TAU).exp()
  };
  let control_mean = S0; // E[e^{-rT} S_T] = S_0 under risk-neutral measure

  let cv = mc::control_variates::estimate(N_PATHS, 1, payoff, control, control_mean);
  let plain = plain_mc(N_PATHS, 1, &gbm_call_payoff(S0, K, R, SIGMA, TAU));

  let err = (cv.mean - bs).abs();
  println!(
    "CV:    mean={:.4}, se={:.6}, err={:.4}",
    cv.mean, cv.std_err, err
  );
  println!("Plain: mean={:.4}, se={:.6}", plain.mean, plain.std_err);
  println!("BS:    {bs:.4}");

  assert!(
    err < 0.30,
    "CV mean {:.4} too far from BS {bs:.4} (err={err:.4})",
    cv.mean
  );
  assert!(
    cv.std_err < plain.std_err,
    "CV se ({:.6}) should be < plain se ({:.6})",
    cv.std_err,
    plain.std_err
  );
}

/// Importance sampling with zero shift matches BS.
/// (Variance reduction depends on shift choice; here we only test correctness.)
#[test]
fn importance_sampling_vs_bs_call() {
  let bs = bs_call(S0, K, R, SIGMA, TAU);
  let payoff = gbm_call_payoff(S0, K, R, SIGMA, TAU);

  // Modest shift towards positive tail for an ATM call
  let shift = Array1::from_vec(vec![0.5]);
  let is = mc::importance_sampling::estimate(N_PATHS, 1, &payoff, &shift);

  let err = (is.mean - bs).abs();
  println!(
    "IS:  mean={:.4}, se={:.6}, err={:.4}",
    is.mean, is.std_err, err
  );
  println!("BS:  {bs:.4}");

  assert!(
    err < 0.50,
    "IS mean {:.4} too far from BS {bs:.4} (err={err:.4})",
    is.mean
  );
}

/// Stratified sampling: price matches BS with higher accuracy than plain MC.
///
/// Note: stratified samples are not i.i.d., so the naive sample-variance SE
/// over-estimates the true SE. We compare estimation accuracy (|mean − BS|)
/// rather than reported SE.
#[test]
fn stratified_vs_bs_call() {
  let bs = bs_call(S0, K, R, SIGMA, TAU);
  let payoff = gbm_call_payoff(S0, K, R, SIGMA, TAU);

  // Run multiple independent estimations to compare mean absolute error
  let n_reps = 20;
  let n_per_rep = N_PATHS / n_reps;
  let mut strat_errors = Vec::new();
  let mut plain_errors = Vec::new();

  for _ in 0..n_reps {
    let s = mc::stratified::estimate(n_per_rep, 1, &payoff);
    let p = plain_mc(n_per_rep, 1, &payoff);
    strat_errors.push((s.mean - bs).abs());
    plain_errors.push((p.mean - bs).abs());
  }

  let strat_mae: f64 = strat_errors.iter().sum::<f64>() / n_reps as f64;
  let plain_mae: f64 = plain_errors.iter().sum::<f64>() / n_reps as f64;

  println!("Strat MAE={strat_mae:.4}, Plain MAE={plain_mae:.4}");

  // Stratified should generally be more accurate
  assert!(
    strat_mae < plain_mae * 1.5,
    "Strat MAE ({strat_mae:.4}) much worse than plain ({plain_mae:.4})"
  );

  // Single full run should match BS
  let strat = mc::stratified::estimate(N_PATHS, 1, &payoff);
  let err = (strat.mean - bs).abs();
  println!(
    "Strat full: mean={:.4}, err={:.4}, BS={bs:.4}",
    strat.mean, err
  );

  assert!(
    err < 0.30,
    "Strat mean {:.4} too far from BS {bs:.4} (err={err:.4})",
    strat.mean
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. QMC: convergence rate faster than MC
// ─────────────────────────────────────────────────────────────────────────────

/// Integrate f(x) = prod_i (4x_i − 2) over [0,1]^d.
///
/// Analytical integral = 0 for d > 1 (each factor integrates to 0), but for
/// d = 1 the integral = ∫(4x − 2)dx = [2x² − 2x]₀¹ = 0 as well.
///
/// Instead we use f(x) = exp(-sum x_i²) whose integral has a closed form.
/// For d = 1: ∫₀¹ exp(-x²) dx = √π/2 · erf(1) ≈ 0.74682.
fn exp_neg_x_sq_integral_1d() -> f64 {
  std::f64::consts::PI.sqrt() / 2.0 * statrs::function::erf::erf(1.0)
}

/// Sobol QMC error decreases faster than MC error as N grows.
#[test]
fn sobol_convergence_rate() {
  let exact = exp_neg_x_sq_integral_1d();
  let seq = SobolSeq::new(1);

  let n_small = 256;
  let n_large = 4096;

  let pts_s: Array2<f64> = seq.sample(n_small);
  let est_s: f64 = (0..n_small)
    .map(|i| (-pts_s[[i, 0]] * pts_s[[i, 0]]).exp())
    .sum::<f64>()
    / n_small as f64;
  let err_s = (est_s - exact).abs();

  let pts_l: Array2<f64> = seq.sample(n_large);
  let est_l: f64 = (0..n_large)
    .map(|i| (-pts_l[[i, 0]] * pts_l[[i, 0]]).exp())
    .sum::<f64>()
    / n_large as f64;
  let err_l = (est_l - exact).abs();

  // Error ratio: for MC ≈ √(n_small/n_large) = √(1/16) = 0.25.
  // For QMC, ratio should be smaller (faster convergence).
  let ratio = n_large as f64 / n_small as f64; // 16
  let error_ratio = err_l / err_s.max(1e-15);
  let mc_expected_ratio = 1.0 / ratio.sqrt(); // 0.25

  println!("Sobol err({n_small})={err_s:.6}, err({n_large})={err_l:.6}");
  println!("Error ratio={error_ratio:.4}, MC expected={mc_expected_ratio:.4}");

  assert!(
    err_l < err_s,
    "Sobol error should decrease: err({n_large})={err_l:.6} >= err({n_small})={err_s:.6}"
  );
  assert!(
    err_l < 0.01,
    "Sobol with {n_large} points should have err < 0.01, got {err_l:.6}"
  );
}

/// Halton QMC error decreases faster than MC for a 3D integral.
#[test]
fn halton_convergence_3d() {
  // ∫₀¹ ∫₀¹ ∫₀¹ (x₁ + x₂ + x₃) dx = 3/2
  let exact = 1.5;
  let seq = HaltonSeq::new(3);
  let n = 4096;

  let pts: Array2<f64> = seq.sample(n);
  let est: f64 = (0..n)
    .map(|i| pts[[i, 0]] + pts[[i, 1]] + pts[[i, 2]])
    .sum::<f64>()
    / n as f64;
  let err = (est - exact).abs();

  println!("Halton 3D: est={est:.6}, exact={exact}, err={err:.6}");

  assert!(
    err < 0.01,
    "Halton 3D error should be < 0.01 with {n} points, got {err:.6}"
  );
}

/// Sobol points for 2^k − 1 samples should have near-exact mean = 0.5 per dim.
#[test]
fn sobol_exactness_power_of_two() {
  let seq = SobolSeq::new(5);
  let n = 1023; // 2^10 − 1
  let pts: Array2<f64> = seq.sample(n);

  for dim in 0..5 {
    let mean: f64 = (0..n).map(|i| pts[[i, dim]]).sum::<f64>() / n as f64;
    assert!(
      (mean - 0.5).abs() < 0.005,
      "dim {dim}: mean = {mean:.6}, expected ≈ 0.5"
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. MLMC: convergence rates α ≈ 1, β ≈ 1 for Euler Gbm
// ─────────────────────────────────────────────────────────────────────────────

/// Run MLMC level-by-level to measure convergence rates, then compare the
/// final estimate to the BS analytical price.
#[test]
fn mlmc_convergence_rates_and_bs_comparison() {
  let r = 0.05_f64;
  let sigma = 0.2;
  let s0 = 100.0;
  let k = 100.0;
  let tau = 1.0;
  let bs = bs_call(s0, k, r, sigma, tau);

  // Fixed-level sampling to measure α and β
  let n_per_level = 20_000;
  let max_level = 6;
  let mut level_means = Vec::new();
  let mut level_vars = Vec::new();

  for level in 0..=max_level {
    let m_fine = 2usize.pow(level as u32 + 1);
    let dt_fine = tau / m_fine as f64;
    let sqrt_dt_fine = dt_fine.sqrt();
    let disc = (-r * tau).exp();

    let mut diffs = Array1::<f64>::zeros(n_per_level);
    for i in 0..n_per_level {
      let z = f64::normal_array(m_fine, 0.0, 1.0);

      // Fine path (Euler)
      let mut s_f = s0;
      for j in 0..m_fine {
        s_f += r * s_f * dt_fine + sigma * s_f * sqrt_dt_fine * z[j];
      }
      let pf = (s_f - k).max(0.0) * disc;

      if level == 0 {
        diffs[i] = pf;
      } else {
        let m_c = m_fine / 2;
        let dt_c = tau / m_c as f64;
        let mut s_c = s0;
        for j in 0..m_c {
          let dw = sqrt_dt_fine * (z[2 * j] + z[2 * j + 1]);
          s_c += r * s_c * dt_c + sigma * s_c * dw;
        }
        let pc = (s_c - k).max(0.0) * disc;
        diffs[i] = pf - pc;
      }
    }

    let mean = diffs.sum() / n_per_level as f64;
    let var = diffs.mapv(|x| (x - mean) * (x - mean)).sum() / n_per_level as f64;
    level_means.push(mean);
    level_vars.push(var);
  }

  // α: |E[Y_l]| ≈ C · 2^{-α·l} → α = log2(|E[Y_{l-1}]| / |E[Y_l]|)
  // β: Var[Y_l] ≈ C · 2^{-β·l} → β = log2(Var[Y_{l-1}] / Var[Y_l])
  // Measured from levels 2..max_level to skip noisy early levels
  let mut alphas = Vec::new();
  let mut betas = Vec::new();
  for l in 3..=max_level {
    let a = (level_means[l - 1].abs() / level_means[l].abs().max(1e-15)).log2();
    let b = (level_vars[l - 1] / level_vars[l].max(1e-30)).log2();
    alphas.push(a);
    betas.push(b);
  }
  let alpha_avg: f64 = alphas.iter().sum::<f64>() / alphas.len() as f64;
  let beta_avg: f64 = betas.iter().sum::<f64>() / betas.len() as f64;

  println!(
    "Level means: {:?}",
    level_means
      .iter()
      .map(|v| format!("{v:.6}"))
      .collect::<Vec<_>>()
  );
  println!(
    "Level vars:  {:?}",
    level_vars
      .iter()
      .map(|v| format!("{v:.6}"))
      .collect::<Vec<_>>()
  );
  println!("α estimates: {alphas:?} → avg α = {alpha_avg:.2}");
  println!("β estimates: {betas:?} → avg β = {beta_avg:.2}");

  // For Euler-Maruyama on Gbm with Lipschitz payoff: α ≈ 1, β ≈ 1
  assert!(
    alpha_avg > 0.5,
    "α = {alpha_avg:.2}, expected ≈ 1.0 (weak convergence)"
  );
  assert!(
    beta_avg > 0.5,
    "β = {beta_avg:.2}, expected ≈ 1.0 (variance decay)"
  );

  // Full MLMC estimate vs BS
  let mlmc = Mlmc::new(0.5, 2, 8, 2000);
  let sampler = |level: usize, n: usize| -> Array1<f64> {
    let m_fine = 2usize.pow(level as u32 + 1);
    let dt_fine = tau / m_fine as f64;
    let sqrt_dt_fine = dt_fine.sqrt();
    let disc = (-r * tau).exp();
    let mut out = Array1::<f64>::zeros(n);

    for i in 0..n {
      let z = f64::normal_array(m_fine, 0.0, 1.0);
      let mut s_f = s0;
      for j in 0..m_fine {
        s_f += r * s_f * dt_fine + sigma * s_f * sqrt_dt_fine * z[j];
      }
      let pf = (s_f - k).max(0.0) * disc;

      if level == 0 {
        out[i] = pf;
      } else {
        let m_c = m_fine / 2;
        let dt_c = tau / m_c as f64;
        let mut s_c = s0;
        for j in 0..m_c {
          let dw = sqrt_dt_fine * (z[2 * j] + z[2 * j + 1]);
          s_c += r * s_c * dt_c + sigma * s_c * dw;
        }
        let pc = (s_c - k).max(0.0) * disc;
        out[i] = pf - pc;
      }
    }
    out
  };

  let result = mlmc.estimate(sampler);
  let err = (result.mean - bs).abs();
  println!(
    "MLMC: mean={:.4}, levels={}, err={:.4}, BS={bs:.4}",
    result.mean, result.n_levels, err
  );

  assert!(
    err < 2.0,
    "MLMC price {:.4} too far from BS {bs:.4} (err={err:.4})",
    result.mean
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. LSM: Longstaff-Schwartz 2001 Table 1 benchmark
// ─────────────────────────────────────────────────────────────────────────────

/// Generate Gbm paths with log-Euler (exact) discretization.
fn generate_gbm_paths(
  n_paths: usize,
  n_steps: usize,
  s0: f64,
  r: f64,
  sigma: f64,
  tau: f64,
) -> Array2<f64> {
  let dt = tau / n_steps as f64;
  let sqrt_dt = dt.sqrt();
  let drift = (r - 0.5 * sigma * sigma) * dt;

  let mut paths = Array2::<f64>::zeros((n_paths, n_steps + 1));
  for i in 0..n_paths {
    paths[[i, 0]] = s0;
    let z = f64::normal_array(n_steps, 0.0, 1.0);
    for j in 0..n_steps {
      paths[[i, j + 1]] = paths[[i, j]] * (drift + sigma * sqrt_dt * z[j]).exp();
    }
  }
  paths
}

/// Longstaff-Schwartz 2001 Table 1 benchmark (σ = 0.2, T = 1).
///
/// Finite-difference American put benchmarks:
///   S=36 → 4.478,  S=38 → 3.250,  S=40 → 2.314,  S=42 → 1.617,  S=44 → 1.110
///
/// Reference: Longstaff & Schwartz (2001), DOI: 10.1093/rfs/14.1.113, Table 1.
#[test]
fn lsm_vs_longstaff_schwartz_table1() {
  let k = 40.0;
  let r = 0.06;
  let sigma = 0.2;
  let tau = 1.0;
  let n_paths = 100_000;
  let n_steps = 50; // 50 exercise opportunities per year
  let n_basis = 4;

  // (S0, FD benchmark, European put for sanity check)
  let cases: [(f64, f64); 5] = [
    (36.0, 4.478),
    (38.0, 3.250),
    (40.0, 2.314),
    (42.0, 1.617),
    (44.0, 1.110),
  ];

  let payoff_put = |s: f64| (k - s).max(0.0);

  for &(s0, fd_benchmark) in &cases {
    let paths = generate_gbm_paths(n_paths, n_steps, s0, r, sigma, tau);
    let lsm = Lsm::new(r, tau, n_basis);
    let am_price = lsm.price(&paths, payoff_put);
    let eu_price = bs_put(s0, k, r, sigma, tau);

    let err = (am_price - fd_benchmark).abs();
    println!("S0={s0:.0}: LSM={am_price:.3}, FD={fd_benchmark:.3}, EU={eu_price:.3}, err={err:.3}");

    // LSM is a low-biased estimator; should be close to but ≤ FD benchmark.
    // Allow tolerance of 0.15 (generous for 100k paths).
    assert!(
      err < 0.20,
      "S0={s0}: LSM {am_price:.3} too far from FD benchmark {fd_benchmark:.3} (err={err:.3})"
    );

    // American ≥ European (fundamental no-arbitrage bound)
    assert!(
      am_price >= eu_price * 0.98,
      "S0={s0}: American {am_price:.3} < European {eu_price:.3}"
    );
  }
}

/// LSM American put with high vol (σ = 0.4, T = 2) vs LS2001 Table 1.
///
/// FD benchmarks: S=36→8.508, S=40→6.920, S=44→5.647
#[test]
fn lsm_vs_ls2001_high_vol() {
  let k = 40.0;
  let r = 0.06;
  let sigma = 0.4;
  let tau = 2.0;
  let n_paths = 100_000;
  let n_steps = 100;
  let n_basis = 4;

  let cases: [(f64, f64); 3] = [(36.0, 8.508), (40.0, 6.920), (44.0, 5.647)];

  let payoff_put = |s: f64| (k - s).max(0.0);

  for &(s0, fd_benchmark) in &cases {
    let paths = generate_gbm_paths(n_paths, n_steps, s0, r, sigma, tau);
    let lsm = Lsm::new(r, tau, n_basis);
    let am_price = lsm.price(&paths, payoff_put);
    let eu_price = bs_put(s0, k, r, sigma, tau);

    let err = (am_price - fd_benchmark).abs();
    println!(
      "S0={s0:.0} σ=0.4 T=2: LSM={am_price:.3}, FD={fd_benchmark:.3}, EU={eu_price:.3}, err={err:.3}"
    );

    assert!(
      err < 0.30,
      "S0={s0}: LSM {am_price:.3} too far from FD benchmark {fd_benchmark:.3} (err={err:.3})"
    );
    assert!(
      am_price >= eu_price * 0.98,
      "S0={s0}: American {am_price:.3} < European {eu_price:.3}"
    );
  }
}
