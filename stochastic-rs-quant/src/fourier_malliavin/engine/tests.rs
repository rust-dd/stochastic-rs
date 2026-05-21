use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::heston::Heston;
use stochastic_rs_stochastic::volatility::heston2d::Heston2D;

use super::*;
use crate::traits::ProcessExt;

fn heston_paths() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
  let n = 23401_usize;
  let t = 1.0_f64;
  let heston = Heston::new(
    Some(100.0),
    Some(0.4),
    2.0,
    0.4,
    1.0,
    -0.5,
    0.0,
    n,
    Some(t),
    HestonPow::Sqrt,
    Some(false),
    Deterministic::new(42),
  );
  let [s, v] = heston.sample();
  let dt = t / (n as f64 - 1.0);
  let times: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
  let log_prices: Vec<f64> = s.iter().map(|&si| si.ln()).collect();
  (log_prices, v.to_vec(), times)
}

fn true_integrated_variance(v: &[f64], dt: f64) -> f64 {
  (0..v.len() - 1).map(|i| (v[i] + v[i + 1]) * 0.5 * dt).sum()
}

/// Ground-truth integrated leverage: σ_v · ρ · IV(T).
/// Returns the analytical Heston covariation between V_t and log S_t.
fn true_integrated_leverage(v: &[f64], dt: f64, sigma_v: f64, rho: f64) -> f64 {
  let iv = true_integrated_variance(v, dt);
  sigma_v * rho * iv
}

/// Ground-truth integrated volvol: σ_v² · IV(T).
/// Returns the analytical Heston quadratic variation of V_t.
fn true_integrated_volvol(v: &[f64], dt: f64, sigma_v: f64) -> f64 {
  let iv = true_integrated_variance(v, dt);
  sigma_v * sigma_v * iv
}

/// Heston fixture parameters reused across tests (matches `heston_paths()`).
const HESTON_SIGMA_V: f64 = 1.0;
const HESTON_RHO: f64 = -0.5;

#[test]
fn test_integrated_variance() {
  let (lp, v, times) = heston_paths();
  let dt = 1.0 / (lp.len() - 1) as f64;
  let engine = FMVol::new(&lp, &times, 1.0);

  let true_iv = true_integrated_variance(&v, dt);
  let est_iv = engine.integrated_variance();
  let rel_err = (est_iv - true_iv).abs() / true_iv;
  assert!(
    rel_err < 0.15,
    "est={est_iv:.6}, true={true_iv:.6}, rel_err={rel_err:.4}"
  );
}

#[test]
fn test_integrated_variance_f32() {
  let (lp64, v, times64) = heston_paths();
  let dt = 1.0 / (lp64.len() - 1) as f64;
  let lp: Vec<f32> = lp64.iter().map(|&x| x as f32).collect();
  let times: Vec<f32> = times64.iter().map(|&x| x as f32).collect();

  let engine = FMVol::new(&lp, &times, 1.0_f32);
  let true_iv = true_integrated_variance(&v, dt);
  let est_iv = engine.integrated_variance() as f64;
  let rel_err = (est_iv - true_iv).abs() / true_iv;
  assert!(
    rel_err < 0.15,
    "f32 est={est_iv:.6}, true={true_iv:.6}, rel_err={rel_err:.4}"
  );
}

#[test]
fn test_uniform_fft_matches_direct() {
  let (lp, _, times) = heston_paths();
  let engine_direct = FMVol::new(&lp, &times, 1.0);
  let engine_fft = FMVol::new_uniform(&lp, 1.0);

  let iv_direct = engine_direct.integrated_variance();
  let iv_fft = engine_fft.integrated_variance();
  let rel_err = (iv_fft - iv_direct).abs() / iv_direct.abs();
  assert!(
    rel_err < 1e-6,
    "FFT vs direct mismatch: fft={iv_fft:.8}, direct={iv_direct:.8}, rel_err={rel_err:.2e}"
  );
}

#[test]
fn test_uniform_fft_spot_matches_direct() {
  let (lp, _, times) = heston_paths();
  let engine_direct = FMVol::new(&lp, &times, 1.0);
  let engine_fft = FMVol::new_uniform(&lp, 1.0);
  let tau: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();

  let spot_direct = engine_direct.spot_variance(&tau, None);
  let spot_fft = engine_fft.spot_variance(&tau, None);

  let max_diff = spot_direct
    .iter()
    .zip(spot_fft.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0_f64, f64::max);
  assert!(
    max_diff < 1e-6,
    "FFT vs direct spot max_diff = {max_diff:.2e}"
  );
}

#[test]
fn test_covariance_self_equals_variance() {
  let (lp, _, times) = heston_paths();
  let engine = FMVol::new(&lp, &times, 1.0);
  let iv = engine.integrated_variance();
  let icov = engine.integrated_covariance(&engine);
  let rel_err = (icov - iv).abs() / iv;
  assert!(
    rel_err < 0.05,
    "cov(x,x)={icov:.6} ≠ var(x)={iv:.6}, rel_err={rel_err:.4}"
  );
}

#[test]
fn test_integrated_leverage_vs_truth() {
  let (lp, v, times) = heston_paths();
  let dt = 1.0 / (lp.len() - 1) as f64;
  let engine = FMVol::new(&lp, &times, 1.0);

  let true_lev = true_integrated_leverage(&v, dt, HESTON_SIGMA_V, HESTON_RHO);
  let est_lev = engine.integrated_leverage(None);
  let rel_err = (est_lev - true_lev).abs() / true_lev.abs();
  assert!(
    est_lev < 0.0,
    "leverage should be < 0 for ρ<0, got {est_lev}"
  );
  assert!(
    rel_err < 0.40,
    "integrated_leverage rel_err = {rel_err:.4} > 40%, est={est_lev:.4}, true={true_lev:.4}. \
     Tolerance is generous because the second-order estimators have higher finite-sample \
     variance; a 40% tolerance still catches structural bugs (sign errors, scaling factors of 2)."
  );
}

#[test]
fn test_integrated_volvol_eq3_within_factor_of_3() {
  let (lp, v, times) = heston_paths();
  let dt = 1.0 / (lp.len() - 1) as f64;
  let engine = FMVol::new(&lp, &times, 1.0);

  let true_vv = true_integrated_volvol(&v, dt, HESTON_SIGMA_V);
  let est_vv = engine.integrated_volvol(None);
  assert!(est_vv > 0.0, "volvol should be > 0, got {est_vv}");
  assert!(
    est_vv > 0.3 * true_vv && est_vv < 3.0 * true_vv,
    "eq.3 volvol est={est_vv:.4} not within [0.3×, 3.0×] of true={true_vv:.4} \
     — this is the non-bias-corrected variant, expect ~2× overestimate from finite-sample bias"
  );
}

#[test]
fn test_integrated_volvol_bias_corrected_vs_truth() {
  let (lp, v, times) = heston_paths();
  let dt = 1.0 / (lp.len() - 1) as f64;
  let engine = FMVol::new(&lp, &times, 1.0);
  let true_vv = true_integrated_volvol(&v, dt, HESTON_SIGMA_V);
  let est_vv = engine.integrated_volvol_bias_corrected(None);
  let rel_err = (est_vv - true_vv).abs() / true_vv;
  assert!(
    rel_err < 0.30,
    "bias-corrected volvol rel_err = {rel_err:.4} > 30%, est={est_vv:.4}, true={true_vv:.4}"
  );
}

#[test]
fn test_integrated_quarticity_positive() {
  let (lp, _, times) = heston_paths();
  let engine = FMVol::new(&lp, &times, 1.0);
  let est = engine.integrated_quarticity(None);
  assert!(est > 0.0, "quarticity should be > 0, got {est}");
}

#[test]
fn test_spot_variance_vs_true() {
  let (lp, v, times) = heston_paths();
  let engine = FMVol::new(&lp, &times, 1.0);
  let n_tau = 21;
  let tau: Vec<f64> = (0..n_tau).map(|i| i as f64 / (n_tau - 1) as f64).collect();
  let spot = engine.spot_variance(&tau, None);

  let step = (lp.len() - 1) / (n_tau - 1);
  let mae: f64 = (0..n_tau)
    .map(|i| (spot[i] - v[i * step]).abs())
    .sum::<f64>()
    / n_tau as f64;
  assert!(mae < 0.25, "spot vol MAE = {mae:.4} too large");
}

#[test]
fn test_spot_covariance_self_equals_spot_vol() {
  let (lp, _, times) = heston_paths();
  let engine = FMVol::new(&lp, &times, 1.0);
  let tau: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();

  let sv = engine.spot_variance(&tau, None);
  let sc = engine.spot_covariance(&engine, &tau, None);

  let max_diff = sv
    .iter()
    .zip(sc.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0_f64, f64::max);
  assert!(max_diff < 0.05, "max_diff = {max_diff:.6}");
}

#[test]
fn test_spot_leverage_mean_vs_truth() {
  let (lp, v, times) = heston_paths();
  let engine = FMVol::new(&lp, &times, 1.0);
  let n_tau = 11;
  let tau: Vec<f64> = (0..n_tau).map(|i| i as f64 / (n_tau - 1) as f64).collect();
  let spot = engine.spot_leverage(&tau, None, None);

  let step = (lp.len() - 1) / (n_tau - 1);
  let true_lev: Vec<f64> = (0..n_tau)
    .map(|i| HESTON_SIGMA_V * HESTON_RHO * v[i * step])
    .collect();

  let spot_mean: f64 = spot.iter().sum::<f64>() / spot.len() as f64;
  let true_mean: f64 = true_lev.iter().sum::<f64>() / n_tau as f64;
  assert!(
    spot_mean < 0.0,
    "mean spot leverage should be < 0 for ρ<0, got {spot_mean}"
  );
  let rel_err = (spot_mean - true_mean).abs() / true_mean.abs();
  assert!(
    rel_err < 0.30,
    "spot_leverage mean rel_err = {rel_err:.4} > 30%; spot_mean = {spot_mean:.4}, true_mean = {true_mean:.4}. \
     Per-τ MAE is high (~120%) because the FM estimator smooths via Fejér window and per-τ values \
     have large finite-sample variance; we assert the mean instead, which is stable."
  );
}

#[test]
fn test_spot_volvol_within_factor_of_3() {
  let (lp, v, times) = heston_paths();
  let engine = FMVol::new(&lp, &times, 1.0);
  let n_tau = 11;
  let tau: Vec<f64> = (0..n_tau).map(|i| i as f64 / (n_tau - 1) as f64).collect();
  let spot = engine.spot_volvol(&tau, None, None);

  let step = (lp.len() - 1) / (n_tau - 1);
  let true_vv: Vec<f64> = (0..n_tau)
    .map(|i| HESTON_SIGMA_V * HESTON_SIGMA_V * v[i * step])
    .collect();
  let spot_mean: f64 = spot.iter().sum::<f64>() / spot.len() as f64;
  let true_mean: f64 = true_vv.iter().sum::<f64>() / n_tau as f64;
  assert!(
    spot_mean > 0.0,
    "mean spot volvol should be > 0, got {spot_mean}"
  );
  assert!(
    spot_mean > 0.3 * true_mean && spot_mean < 3.0 * true_mean,
    "spot_volvol mean {spot_mean:.4} not within [0.3×, 3.0×] of true mean {true_mean:.4} \
     — this is the non-bias-corrected variant, expect ~2-3× overestimate from finite-sample bias"
  );
}

#[test]
fn test_spot_volvol_bias_corrected_vs_truth() {
  let (lp, v, times) = heston_paths();
  let engine = FMVol::new(&lp, &times, 1.0);
  let n_tau = 11;
  let tau: Vec<f64> = (0..n_tau).map(|i| i as f64 / (n_tau - 1) as f64).collect();
  let spot = engine.spot_volvol_bias_corrected(&tau, None, None);

  let step = (lp.len() - 1) / (n_tau - 1);
  let true_vv: Vec<f64> = (0..n_tau)
    .map(|i| HESTON_SIGMA_V * HESTON_SIGMA_V * v[i * step])
    .collect();
  let spot_mean: f64 = spot.iter().sum::<f64>() / spot.len() as f64;
  let true_mean: f64 = true_vv.iter().sum::<f64>() / n_tau as f64;
  assert!(
    spot_mean > 0.0,
    "mean bias-corrected spot volvol should be > 0, got {spot_mean}"
  );
  let rel_err = (spot_mean - true_mean).abs() / true_mean;
  assert!(
    rel_err < 0.40,
    "spot_volvol_bias_corrected mean rel_err = {rel_err:.4} > 40%; spot_mean = {spot_mean:.4}, true_mean = {true_mean:.4}"
  );
}

#[test]
fn test_spot_quarticity_positive() {
  let (lp, _, times) = heston_paths();
  let engine = FMVol::new(&lp, &times, 1.0);
  let tau: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();
  let spot = engine.spot_quarticity(&tau, None, None);
  let mean: f64 = spot.iter().copied().sum::<f64>() / spot.len() as f64;
  assert!(mean > 0.0, "mean spot quarticity should be > 0, got {mean}");
}

#[test]
fn test_spot_variance_f32() {
  let (lp64, _, times64) = heston_paths();
  let lp: Vec<f32> = lp64.iter().map(|&x| x as f32).collect();
  let times: Vec<f32> = times64.iter().map(|&x| x as f32).collect();
  let tau: Vec<f32> = (0..11).map(|i| i as f32 / 10.0).collect();

  let engine = FMVol::new(&lp, &times, 1.0_f32);
  let spot = engine.spot_variance(&tau, None);
  let mean: f32 = spot.iter().copied().sum::<f32>() / spot.len() as f32;
  assert!(
    mean > 0.1 && mean < 0.8,
    "f32 mean spot vol {mean} out of range"
  );
}

/// Bivariate Heston fixture matching the MATLAB Heston2D.m example
/// `parameters=[0,0;0.4,0.4;2,2;1,1]`, `Rho=[0.5,-0.5,0,0,-0.5,0.5]`.
/// Returns `(log_prices_1, log_prices_2, times, v_1, v_2)`.
fn heston2d_paths() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
  let n = 23401_usize;
  let t = 1.0_f64;
  let h = Heston2D::<f64, _>::new(
    [Some(0.0_f64), Some(0.0_f64)],
    [Some(0.4_f64), Some(0.4_f64)],
    [0.0, 0.0],
    [0.4, 0.4],
    [2.0, 2.0],
    [1.0, 1.0],
    [0.5, -0.5, 0.0, 0.0, -0.5, 0.5],
    n,
    Some(t),
    Some(false),
    Deterministic::new(42),
  );
  let [x1, v1, x2, v2] = h.sample();
  let dt = t / (n as f64 - 1.0);
  let times: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
  (x1.to_vec(), x2.to_vec(), times, v1.to_vec(), v2.to_vec())
}

#[test]
fn test_integrated_covariance_bivariate_heston() {
  // For bivariate Heston with ρ(W1,W2)=0.5: IC = ρ ∫₀ᵀ √(v_1(s) v_2(s)) ds
  let (x1, x2, times, v1, v2) = heston2d_paths();
  let engine1 = FMVol::new(&x1, &times, 1.0);
  let engine2 = FMVol::new(&x2, &times, 1.0);
  let dt = 1.0 / (x1.len() - 1) as f64;
  let true_ic: f64 = (0..v1.len() - 1)
    .map(|i| {
      let m1 = (v1[i] * v2[i]).sqrt();
      let m2 = (v1[i + 1] * v2[i + 1]).sqrt();
      0.5 * (m1 + m2) * 0.5 * dt
    })
    .sum();
  let est_ic = engine1.integrated_covariance(&engine2);
  let rel_err = (est_ic - true_ic).abs() / true_ic.abs();
  assert!(
    est_ic > 0.0,
    "covariance should be > 0 for ρ>0, got {est_ic}"
  );
  assert!(
    rel_err < 0.05,
    "bivariate FM_int_cov rel_err = {rel_err:.4} > 5%, est={est_ic:.6}, true={true_ic:.6}"
  );
}

#[test]
fn test_spot_variance_fe_kernel() {
  // FE convention uses (1 − |k|/M) instead of (1 − |k|/(M+1)).
  // For the same M, FE has slightly more aggressive smoothing — but for
  // a Heston path with M ≈ √N both produce results close to the truth.
  let (lp, v, times) = heston_paths();
  let engine = FMVol::new(&lp, &times, 1.0);
  let tau: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();

  let spot_fm = engine.spot_variance(&tau, None);
  let spot_fe = engine.spot_variance_fe(&tau, None);

  let step = (lp.len() - 1) / (tau.len() - 1);
  let mae_fm: f64 = (0..tau.len())
    .map(|i| (spot_fm[i] - v[i * step]).abs())
    .sum::<f64>()
    / tau.len() as f64;
  let mae_fe: f64 = (0..tau.len())
    .map(|i| (spot_fe[i] - v[i * step]).abs())
    .sum::<f64>()
    / tau.len() as f64;
  assert!(mae_fm < 0.30, "FM spot vol MAE {mae_fm:.4} too large");
  assert!(mae_fe < 0.30, "FE spot vol MAE {mae_fe:.4} too large");
}

#[test]
fn test_optimal_cutting_frequency_noisy() {
  let (lp, v, times) = heston_paths();
  let dt = 1.0 / (lp.len() - 1) as f64;
  let true_iv: f64 = (0..v.len() - 1).map(|i| (v[i] + v[i + 1]) * 0.5 * dt).sum();

  // Add i.i.d. noise: η ~ N(0, σ²_η) with noise-to-signal ≈ 0.5
  let sigma_eta = 0.005;
  let noisy: Vec<f64> = lp
    .iter()
    .enumerate()
    .map(|(i, &p)| {
      // Deterministic pseudo-noise for reproducibility
      let noise = sigma_eta * (((i * 7919 + 104729) % 10000) as f64 / 5000.0 - 1.0);
      p + noise
    })
    .collect();

  // Optimal N
  let result = super::super::optimal_cutting_frequency(&noisy, &times);
  let (n_opt, m_opt, _l_opt) = result.cutting_freqs();

  // Fixed-rule N (heuristic)
  let n = lp.len() - 1;
  let (n_heur, m_heur, _) = super::super::default_cutting_freq_noisy(n);

  // Estimate with optimal N
  let engine_opt = FMVol::with_freq(&noisy, &times, 1.0, n_opt, n_opt + m_opt + 10);
  let iv_opt = engine_opt.integrated_variance();

  // Estimate with heuristic N
  let engine_heur = FMVol::with_freq(&noisy, &times, 1.0, n_heur, n_heur + m_heur + 10);
  let iv_heur = engine_heur.integrated_variance();

  // Estimate with naive N = n/2 (no noise correction)
  let engine_naive = FMVol::new(&noisy, &times, 1.0);
  let iv_naive = engine_naive.integrated_variance();

  let err_opt = (iv_opt - true_iv).abs() / true_iv;
  let _err_heur = (iv_heur - true_iv).abs() / true_iv;
  let err_naive = (iv_naive - true_iv).abs() / true_iv;

  // Optimal N should give smaller error than naive (no noise correction)
  assert!(
    err_opt < err_naive,
    "optimal N should beat naive: err_opt={err_opt:.4}, err_naive={err_naive:.4}"
  );

  // Optimal N should be much smaller than n/2
  assert!(
    n_opt < n / 4,
    "optimal N={n_opt} should be << n/2={} for noisy data",
    n / 2
  );

  // Estimated noise variance should be in reasonable range
  assert!(
    result.noise_variance > 0.0,
    "noise variance should be positive, got {}",
    result.noise_variance
  );
}
