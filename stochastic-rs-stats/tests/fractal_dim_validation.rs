use ndarray::Array1;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stats::fd::FractalDim;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::process::fbm::Fbm;
use stochastic_rs_stochastic::traits::ProcessExt;

#[test]
fn fbm_fractal_dimension_matches_theory() {
  let h = 0.72_f64;
  let d_theory = 2.0 - h;
  let n = 4096_usize;
  let m = 160_usize;
  let kmax = 32_usize;
  let fbm = Fbm::<f64>::new(h, n, Some(1.0), Unseeded);

  let mut d_vario_sum = 0.0;
  let mut d_higuchi_sum = 0.0;
  for _ in 0..m {
    let x = fbm.sample();
    let fd = FractalDim::new(x);
    d_vario_sum += fd.variogram(Some(2.0)).expect("variogram on fBM path");
    d_higuchi_sum += fd.higuchi_fd(kmax).expect("Higuchi on fBM path");
  }
  let d_vario = d_vario_sum / m as f64;
  let d_higuchi = d_higuchi_sum / m as f64;

  assert!(
    (d_vario - d_theory).abs() < 0.05,
    "variogram FD mismatch: D_est={d_vario}, D={d_theory}"
  );
  assert!(
    (d_higuchi - d_theory).abs() < 0.05,
    "higuchi FD mismatch: D_est={d_higuchi}, D={d_theory}"
  );
}

#[test]
fn fbm_hurst_and_fractal_dimension_from_fgn_increments() {
  let h = 0.78_f64;
  let n = 4096_usize;
  let t = 1.0_f64;
  let m = 240_usize;
  let kmax = 32_usize;

  let fgn = Fgn::<f64>::new(h, n, Some(t), Unseeded);
  let mut endpoints = Vec::with_capacity(m);
  let mut d_vario_sum = 0.0_f64;
  let mut d_higuchi_sum = 0.0_f64;
  let mut d_higuchi_count = 0usize;

  for path_idx in 0..m {
    let inc = ProcessExt::sample(&fgn);
    let mut fbm = vec![0.0_f64; n + 1];
    for i in 0..n {
      fbm[i + 1] = fbm[i] + inc[i];
    }
    endpoints.push(fbm[n]);

    let fd = FractalDim::new(Array1::from_vec(fbm));
    d_vario_sum += fd.variogram(Some(2.0)).expect("variogram on fBM path");

    if path_idx % 2 == 0 {
      d_higuchi_sum += fd.higuchi_fd(kmax).expect("Higuchi on fBM path");
      d_higuchi_count += 1;
    }
  }

  let fractal_dim_vario = d_vario_sum / m as f64;
  let fractal_dim_higuchi = d_higuchi_sum / d_higuchi_count as f64;
  let h_from_vario = 2.0 - fractal_dim_vario;
  let fractal_dim_theory = 2.0 - h;

  let endpoint_mean = endpoints.iter().sum::<f64>() / endpoints.len() as f64;
  let endpoint_var = endpoints
    .iter()
    .map(|x| {
      let d = *x - endpoint_mean;
      d * d
    })
    .sum::<f64>()
    / endpoints.len() as f64;

  assert!(
    (h_from_vario - h).abs() < 0.05,
    "H mismatch from variogram FD: h_est={h_from_vario}, h={h}"
  );
  assert!(
    (fractal_dim_vario - fractal_dim_theory).abs() < 0.05,
    "variogram FD mismatch: D_est={fractal_dim_vario}, D={fractal_dim_theory}"
  );
  assert!(
    (fractal_dim_higuchi - fractal_dim_theory).abs() < 0.05,
    "Higuchi FD mismatch: D_est={fractal_dim_higuchi}, D={fractal_dim_theory}"
  );
  assert!(
    ((endpoint_var / (t.powf(2.0 * h))) - 1.0).abs() < 0.18,
    "endpoint variance mismatch: emp={endpoint_var}, theory={}",
    t.powf(2.0 * h)
  );
}
