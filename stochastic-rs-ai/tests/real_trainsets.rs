//! The upstream training sets in `tests/data`, which the published package leaves out.

use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use ndarray::Array2;
use ndarray::s;
use stochastic_rs_ai::Device;
use stochastic_rs_ai::volatility::common::TrainConfig;
use stochastic_rs_ai::volatility::common::TrainReport;
use stochastic_rs_ai::volatility::common::load_trainset_gzip_npy;
use stochastic_rs_ai::volatility::heston;
use stochastic_rs_ai::volatility::heston::HestonNn;
use stochastic_rs_ai::volatility::one_factor;
use stochastic_rs_ai::volatility::one_factor::OneFactorNn;
use stochastic_rs_ai::volatility::rbergomi;
use stochastic_rs_ai::volatility::rbergomi::RBergomiNn;

fn data(file: &str) -> PathBuf {
  Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests/data")
    .join(file)
}

fn assert_set_trains(
  file: &str,
  input_dim: usize,
  output_dim: usize,
  (lb, ub): (&[f32], &[f32]),
  train: impl FnOnce(&Array2<f32>, &Array2<f32>, &TrainConfig) -> Result<TrainReport>,
) -> Result<()> {
  let (params, surfaces) = load_trainset_gzip_npy(data(file), input_dim, output_dim, None)?;
  for j in 0..input_dim {
    let range = (lb[j] - 1e-6)..=(ub[j] + 1e-6);
    assert!(
      params.column(j).iter().all(|v| range.contains(v)),
      "parameter {j} leaves its training box"
    );
  }
  assert!(surfaces.iter().all(|v| v.is_finite() && *v > 0.0));

  let cfg = TrainConfig {
    epochs: 10,
    batch_size: 64,
    ..TrainConfig::default()
  };
  let head = s![..2_000, ..];
  let report = train(
    &params.slice(head).to_owned(),
    &surfaces.slice(head).to_owned(),
    &cfg,
  )?;
  let first = report.epochs[0].val_rmse;
  let last = report.epochs[report.epochs.len() - 1].val_rmse;
  assert!(
    last < 0.5 * first && last < 0.4,
    "validation RMSE {first} -> {last}"
  );
  Ok(())
}

#[test]
fn heston_set_trains() -> Result<()> {
  assert_set_trains(
    "HestonTrainSet.txt.gz",
    heston::INPUT_DIM,
    heston::OUTPUT_DIM,
    (&heston::PARAM_LB, &heston::PARAM_UB),
    |p, s, c| HestonNn::new(&Device::Cpu)?.train(p, s, c),
  )
}

#[test]
fn one_factor_set_trains() -> Result<()> {
  assert_set_trains(
    "Bergomi1FactorTrainSet.txt.gz",
    one_factor::INPUT_DIM,
    one_factor::OUTPUT_DIM,
    (&one_factor::PARAM_LB, &one_factor::PARAM_UB),
    |p, s, c| OneFactorNn::new(&Device::Cpu)?.train(p, s, c),
  )
}

#[test]
fn rbergomi_set_trains() -> Result<()> {
  assert_set_trains(
    "rBergomiTrainSet.txt.gz",
    rbergomi::INPUT_DIM,
    rbergomi::OUTPUT_DIM,
    (&rbergomi::PARAM_LB, &rbergomi::PARAM_UB),
    |p, s, c| RBergomiNn::new(&Device::Cpu)?.train(p, s, c),
  )
}

#[cfg(feature = "quant")]
#[test]
fn heston_set_is_the_fourier_surface_on_inverse_strikes() -> Result<()> {
  use stochastic_rs_ai::volatility::grid;
  use stochastic_rs_quant::pricing::heston::HestonPricer;
  use stochastic_rs_quant::vol_surface::ModelSurface;

  let (params, surfaces) = load_trainset_gzip_npy(
    data("HestonTrainSet.txt.gz"),
    heston::INPUT_DIM,
    heston::OUTPUT_DIM,
    None,
  )?;
  let n_k = heston::STRIKES.len();
  let ascending = heston::STRIKES.iter().rev().copied().collect::<Vec<f64>>();
  let (mut on_inverse, mut on_direct) = (Vec::new(), Vec::new());
  for r in (0..params.nrows()).step_by(params.nrows() / 30) {
    let p = params.row(r).mapv(f64::from);
    let pricer = HestonPricer::new(p[0], p[1], p[4], p[3], p[2], None);
    let inverse = pricer.vol_surface(1.0, 0.0, 0.0, &ascending, &grid::MATURITIES);
    let direct = pricer.vol_surface(1.0, 0.0, 0.0, &grid::MONEYNESS, &grid::MATURITIES);
    for t in 0..grid::MATURITIES.len() {
      for i in 0..n_k {
        let want = f64::from(surfaces[[r, t * n_k + i]]);
        on_inverse.push((inverse.ivs[[t, n_k - 1 - i]] - want).abs());
        on_direct.push((direct.ivs[[t, i]] - want).abs());
      }
    }
  }
  let median = |mut v: Vec<f64>| {
    v.retain(|x| x.is_finite());
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
  };
  let (inverse, direct) = (median(on_inverse), median(on_direct));
  assert!(inverse < 1e-4, "median |dIV| on inverse strikes: {inverse}");
  assert!(
    direct > 10.0 * inverse,
    "direct strikes: {direct} vs inverse {inverse}"
  );
  Ok(())
}
