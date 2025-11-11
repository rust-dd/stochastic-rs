use std::vec::IntoIter;

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_datasets::{batcher::IterResult2, Batcher};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array1};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;

use crate::stochastic::{diffusion::fou::FOU, noise::fgn::FGN, SamplingExt};

pub fn test_vasicek_1_d(
  epoch_size: usize,
  batch_size: usize,
  n: usize,
  device: &Device,
) -> Result<(
  Batcher<IterResult2<IntoIter<Result<(Tensor, Tensor), candle_core::Error>>>>,
  Vec<f64>,
)> {
  let mut paths = Vec::with_capacity(epoch_size);
  let mu = 2.8;
  let sigma = 1.0;
  let thetas = Array1::random(epoch_size, Uniform::new(0.0, 10.0)).to_vec();
  let hursts = Array1::random(epoch_size, Uniform::new(0.01, 0.99)).to_vec();
  let progress_bar = ProgressBar::new(epoch_size as u64);
  progress_bar.set_style(
    ProgressStyle::with_template(
      "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] ({eta})",
    )?
    .progress_chars("#>-"),
  );
  for idx in 0..epoch_size {
    let hurst = hursts[idx];
    let theta = thetas[idx];
    let fou = FOU::new(
      theta,
      mu,
      sigma,
      n,
      Some(0.0),
      Some(16.0),
      None,
      FGN::<f64>::new(hurst, n - 1, Some(1.0), None),
    );
    let mut path = fou.sample();
    let mean = path.mean().unwrap();
    let std = path.std(0.0);
    path = (path - mean) / std;

    paths.push(Ok((
      Tensor::from_iter(path, device)?,
      Tensor::new(&[thetas[idx]], device)?,
    )));
    progress_bar.inc(1);
  }
  progress_bar.finish();

  let batcher = Batcher::new_r2(paths.into_iter())
    .batch_size(batch_size)
    .return_last_incomplete_batch(false);

  Ok((batcher, hursts))
}

pub fn test_vasicek_2_d(
  epoch_size: usize,
  batch_size: usize,
  n: usize,
  device: &Device,
) -> Result<(
  Batcher<IterResult2<IntoIter<Result<(Tensor, Tensor), candle_core::Error>>>>,
  Vec<f64>,
)> {
  let mut paths = Vec::with_capacity(epoch_size);
  let mu = 2.8;
  let sigma = 1.0;
  let thetas = Array1::random(epoch_size, Uniform::new(0.0, 10.0)).to_vec();
  let hursts = Array1::random(epoch_size, Uniform::new(0.01, 0.99)).to_vec();
  let progress_bar = ProgressBar::new(epoch_size as u64);
  progress_bar.set_style(
    ProgressStyle::with_template(
      "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] ({eta})",
    )?
    .progress_chars("#>-"),
  );
  for idx in 0..epoch_size {
    let hurst = hursts[idx];
    let theta = thetas[idx];
    let fou = FOU::new(
      theta,
      mu,
      sigma,
      n,
      Some(0.0),
      Some(16.0),
      None,
      FGN::<f64>::new(hurst, n - 1, Some(1.0), None),
    );
    let mut path = fou.sample();
    let mean = path.mean().unwrap();
    let std = path.std(0.0);
    path = (path - mean) / std;

    let diff = &path.slice(s![1..]) - &path.slice(s![..-1]);
    let path = path.slice(s![..-1]);
    let paired = path.iter().zip(diff.iter()).collect::<Vec<_>>();
    let paired_tensors = paired
      .iter()
      .map(|pair| {
        let (x, y) = *pair;
        Tensor::new(&[*x, *y], device).unwrap()
      })
      .collect::<Vec<_>>();

    paths.push(Ok((
      Tensor::stack(&paired_tensors, 0)?,
      Tensor::new(&[thetas[idx]], device)?,
    )));
    progress_bar.inc(1);
  }
  progress_bar.finish();

  let batcher = Batcher::new_r2(paths.into_iter())
    .batch_size(batch_size)
    .return_last_incomplete_batch(false);

  Ok((batcher, hursts))
}
