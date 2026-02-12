//! # Stochastic Process Simulation Modules
//!
//! `stochastic` provides a modular framework for simulating and analyzing a wide range of stochastic processes.
//! It is designed for high performance, parallelism, and optional GPU acceleration via CUDA.
//!
//! ## Modules
//!
//! | Module            | Description                                                                                                   |
//! |-------------------|---------------------------------------------------------------------------------------------------------------|
//! | [`autoregressive`] | Implements autoregressive stochastic models.                                                                 |
//! | [`diffusion`]      | Handles diffusion processes such as Brownian motion and geometric Brownian motion.                           |
//! | [`interest`]       | Simulates stochastic interest rates using models like Cox-Ingersoll-Ross (CIR).                              |
//! | [`jump`]           | Implements jump processes like Poisson or Bates model, useful in financial mathematics.                      |
//! | [`malliavin`]      | Tools for computing derivatives of stochastic processes (via Malliavin calculus). Optional feature `malliavin`. |
//! | [`noise`]          | Provides Gaussian and fractional Gaussian noise generators.                                                  |
//! | [`process`]        | General abstraction layer for sampling and managing stochastic processes.                                     |
//! | [`volatility`]     | Models stochastic volatility processes such as the Heston model.                                              |
//! | [`sde`]            | Solves stochastic differential equations with multiple integration schemes.                                   |
//! | [`ito`]            | Defines the Itô calculus-based framework for process analysis.                                               |
//! | [`isonormal`]      | Provides isonormal Gaussian processes.                                                                       |
//!
//! ## Features
//!
//! - `cuda`: Enables GPU-based sampling implementations via CUDA
//! - `malliavin`: Enables Malliavin derivative-related functionality
//!
//! ## Parallelism
//!
//! All `sample_par()` methods use `rayon` for parallel execution over samples. Set the `m()` field to define the number of parallel trajectories.
//!
//! ## Example Usage
//!
//! ```rust
//! use stochastic::diffusion::BrownianMotion;
//! use stochastic::SamplingExt;
//!
//! let bm = BrownianMotion::new(0.0, 1.0, 1.0, 1000);
//! let path = bm.sample();
//! ```
//!
//! ## GPU Acceleration
//!
//! When the `cuda` feature is enabled, `sample_cuda()` can be used for faster batch sampling on supported devices.

pub mod autoregressive;
pub mod diffusion;
pub mod interest;
pub mod isonormal;
pub mod ito;
pub mod jump;
pub mod malliavin;
pub mod noise;
pub mod process;
pub mod sde;
pub mod sheet;
pub mod volatility;

use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::SubAssign;

#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use either::Either;
use ndarray::parallel::prelude::*;
use ndarray::Array1;
#[cfg(feature = "cuda")]
use ndarray::Array2;
use ndarray::ScalarOperand;
use ndarray_rand::RandomExt;
use ndrustfft::Zero;
use num_complex::Complex64;

use crate::distributions::normal::SimdNormal;
use crate::distributions::SimdFloatExt;

/// Default number of time steps
pub const N: usize = 1000;
/// Default initial value
pub const X0: f64 = 0.5;
/// Default spot price for financial models
pub const S0: f64 = 100.0;
/// Default strike price
pub const K: f64 = 100.0;

pub trait FloatExt:
  num_traits::Float
  + num_traits::FromPrimitive
  + num_traits::Signed
  + num_traits::FloatConst
  + Sum
  + SimdFloatExt
  + Zero
  + Default
  + Debug
  + Send
  + Sync
  + ScalarOperand
  + AddAssign
  + SubAssign
  + 'static
{
  fn from_usize_(n: usize) -> Self;
  fn normal_array(n: usize, mean: Self, std_dev: Self) -> Array1<Self>;
}

impl FloatExt for f64 {
  fn from_usize_(n: usize) -> Self {
    n as f64
  }

  fn normal_array(n: usize, mean: Self, std_dev: Self) -> Array1<Self> {
    Array1::random(n, SimdNormal::<f64, 64>::new(mean, std_dev))
  }
}

impl FloatExt for f32 {
  fn from_usize_(n: usize) -> Self {
    n as f32
  }

  fn normal_array(n: usize, mean: Self, std_dev: Self) -> Array1<Self> {
    Array1::random(n, SimdNormal::<f32, 64>::new(mean, std_dev))
  }
}

pub trait ProcessExt<T: FloatExt>: Send + Sync {
  type Output: Send;

  fn sample(&self) -> Self::Output;

  // fn sample_with_rng(&self, rng: &mut impl Rng) -> Self::Output;

  fn sample_par(&self, m: usize) -> Vec<Self::Output> {
    (0..m).into_par_iter().map(|_| self.sample()).collect()
  }

  #[cfg(feature = "cuda")]
  fn sample_cuda(&self, _m: usize) -> Result<Either<Array1<f64>, Array2<f64>>> {
    anyhow::bail!("CUDA sampling is not supported for this process")
  }
}

/// Provides analytical functions of the distribution (pdf, cdf, etc)
pub trait DistributionExt {
  /// Characteristic function of the distribution
  fn characteristic_function(&self, _t: f64) -> Complex64 {
    Complex64::new(0.0, 0.0)
  }

  /// Probability density function of the distribution
  fn pdf(&self, _x: f64) -> f64 {
    0.0
  }

  /// Cumulative distribution function of the distribution
  fn cdf(&self, _x: f64) -> f64 {
    0.0
  }

  /// Inverse cumulative distribution function of the distribution
  fn inv_cdf(&self, _p: f64) -> f64 {
    0.0
  }

  /// Mean of the distribution
  fn mean(&self) -> f64 {
    0.0
  }

  /// Median of the distribution
  fn median(&self) -> f64 {
    0.0
  }

  /// Mode of the distribution
  fn mode(&self) -> f64 {
    0.0
  }

  /// Variance of the distribution
  fn variance(&self) -> f64 {
    0.0
  }

  /// Skewness of the distribution
  fn skewness(&self) -> f64 {
    0.0
  }

  /// Kurtosis of the distribution
  fn kurtosis(&self) -> f64 {
    0.0
  }

  /// Entropy of the distribution
  fn entropy(&self) -> f64 {
    0.0
  }

  /// Moment generating function of the distribution
  fn moment_generating_function(&self, _t: f64) -> f64 {
    0.0
  }
}
