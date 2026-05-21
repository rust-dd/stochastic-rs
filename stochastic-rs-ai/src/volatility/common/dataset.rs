use std::fs::File;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use flate2::read::GzDecoder;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

pub fn load_trainset_gzip_npy<P: AsRef<Path>>(
  path: P,
  input_dim: usize,
  output_dim: usize,
  max_rows: Option<usize>,
) -> Result<(Array2<f32>, Array2<f32>)> {
  let path = path.as_ref();
  let file =
    File::open(path).with_context(|| format!("failed to open dataset file at {:?}", path))?;
  let decoder = GzDecoder::new(file);
  let data = Array2::<f64>::read_npy(decoder)
    .with_context(|| format!("failed to decode/load npy data from {:?}", path))?;

  let expected_min_cols = input_dim + output_dim;
  if data.ncols() < expected_min_cols {
    bail!(
      "dataset at {:?} has {} columns, expected at least {} (input_dim + output_dim)",
      path,
      data.ncols(),
      expected_min_cols
    );
  }

  let rows = max_rows.unwrap_or(data.nrows()).min(data.nrows());
  if rows < 2 {
    bail!("dataset at {:?} must contain at least 2 rows", path);
  }

  let params_f64 = data.slice(ndarray::s![0..rows, 0..input_dim]).to_owned();
  let surfaces_f64 = data
    .slice(ndarray::s![0..rows, input_dim..(input_dim + output_dim)])
    .to_owned();

  let params = params_f64.mapv(|v| v as f32);
  let surfaces = surfaces_f64.mapv(|v| v as f32);
  Ok((params, surfaces))
}

pub fn rmse_1d(actual: &[f32], predicted: &[f32]) -> Result<f32> {
  if actual.len() != predicted.len() {
    bail!(
      "rmse_1d length mismatch: {} vs {}",
      actual.len(),
      predicted.len()
    );
  }
  if actual.is_empty() {
    bail!("rmse_1d cannot be computed on empty slices");
  }
  let mse = actual
    .iter()
    .zip(predicted.iter())
    .map(|(a, p)| {
      let d = *a - *p;
      d * d
    })
    .sum::<f32>()
    / (actual.len() as f32);
  Ok(mse.sqrt())
}

pub(super) fn train_test_split_indices(
  n: usize,
  test_ratio: f32,
  seed: u64,
) -> (Vec<usize>, Vec<usize>) {
  let mut idx = (0..n).collect::<Vec<usize>>();
  let mut rng = StdRng::seed_from_u64(seed);
  idx.shuffle(&mut rng);

  let mut n_test = ((n as f32) * test_ratio).round() as usize;
  n_test = n_test.clamp(1, n.saturating_sub(1));
  let test = idx[..n_test].to_vec();
  let train = idx[n_test..].to_vec();
  (train, test)
}

#[cfg(test)]
pub(crate) fn synthetic_surface_dataset(
  lb: &[f32],
  ub: &[f32],
  samples: usize,
  output_dim: usize,
  seed: u64,
) -> (Array2<f32>, Array2<f32>) {
  use rand::Rng;

  let dim = lb.len();
  let mut rng = StdRng::seed_from_u64(seed);
  let mut params = Array2::<f32>::zeros((samples, dim));
  let mut surfaces = Array2::<f32>::zeros((samples, output_dim));

  for i in 0..samples {
    for j in 0..dim {
      let u = rng.random::<f32>();
      params[[i, j]] = lb[j] + (ub[j] - lb[j]) * u;
    }
    for k in 0..output_dim {
      let mut v = 0.2 + 0.03 * (k as f32) / (output_dim as f32);
      for j in 0..dim {
        let center = 0.5 * (lb[j] + ub[j]);
        let half = 0.5 * (ub[j] - lb[j]);
        let x = (params[[i, j]] - center) / half;
        let w = 0.08 + 0.02 * ((j + 1) as f32);
        v += w * x * ((k as f32 + 1.0) * (j as f32 + 1.0) * 0.11).sin();
        v += 0.03 * x * x / ((j + 1) as f32);
      }
      surfaces[[i, k]] = v;
    }
  }

  (params, surfaces)
}
