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
#[cfg(feature = "malliavin")]
pub mod malliavin;
pub mod noise;
pub mod process;
pub mod sde;
pub mod sheet;
pub mod volatility;

use std::fmt::Debug;
use std::ops::AddAssign;
use std::sync::Arc;
use std::sync::Mutex;

#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use either::Either;
use ndarray::parallel::prelude::*;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::ScalarOperand;
use ndarray_rand::RandomExt;
use ndrustfft::Zero;
use num_complex::Complex64;
use rand_distr::Normal;

/// Default number of time steps
pub const N: usize = 1000;
/// Default initial value
pub const X0: f64 = 0.5;
/// Default spot price for financial models
pub const S0: f64 = 100.0;
/// Default strike price
pub const K: f64 = 100.0;

pub trait Float:
  num_traits::Float + Zero + Default + Debug + Send + Sync + ScalarOperand + AddAssign + 'static
{
  fn from_usize(v: usize) -> Self;
  fn normal_array(n: usize, mean: Self, std_dev: Self) -> Array1<Self>;
}

impl Float for f64 {
  fn from_usize(v: usize) -> Self {
    v as f64
  }

  fn normal_array(n: usize, mean: Self, std_dev: Self) -> Array1<Self> {
    Array1::random(n, Normal::new(mean, std_dev).unwrap())
  }
}

impl Float for f32 {
  fn from_usize(v: usize) -> Self {
    v as f32
  }

  fn normal_array(n: usize, mean: Self, std_dev: Self) -> Array1<Self> {
    Array1::random(n, Normal::new(mean, std_dev).unwrap())
  }
}

pub trait Process<T: Float>: Send + Sync {
  type Output: Send;

  fn sample(&self) -> Self::Output;

  fn n(&self) -> usize;

  fn sample_par(&self, m: usize) -> Vec<Self::Output> {
    (0..m).into_par_iter().map(|_| self.sample()).collect()
  }
}

/// Trait for 1D sampling of stochastic processes
pub trait SamplingExt<T: Clone + Send + Sync + Zero>: Send + Sync {
  /// Sample the process
  fn sample(&self) -> Array1<T> {
    unimplemented!()
  }

  /// Sample the process with simd acceleration
  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<T> {
    unimplemented!()
  }

  /// Sample the process with CUDA support
  #[cfg(feature = "cuda")]
  fn sample_cuda(&self) -> Result<Either<Array1<T>, Array2<T>>> {
    unimplemented!()
  }

  /// Parallel sampling
  fn sample_par(&self) -> Array2<T> {
    if self.m().is_none() {
      panic!("m must be specified for parallel sampling");
    }

    let mut xs = Array2::zeros((self.m().unwrap(), self.n()));

    xs.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut x| {
      x.assign(&self.sample());
    });

    xs
  }

  /// Number of time steps
  fn n(&self) -> usize;

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize>;

  /// Distribution of the process
  fn distribution(&mut self) {}

  /// Malliavin derivative of the process
  #[cfg(feature = "malliavin")]
  fn malliavin(&self) -> Array1<T> {
    unimplemented!()
  }

  /// Set CUDA support
  #[cfg(feature = "cuda")]
  fn set_cuda(&mut self, cuda: bool) {}
}

/// Trait for sampling vector-valued stochastic processes
pub trait SamplingVExt<T: Clone + Send + Sync + Zero>: Send + Sync {
  /// Sample the vector process
  fn sample(&self) -> Array2<T>;

  /// Parallel sampling
  fn sample_par(&self) -> Array2<T> {
    unimplemented!()
  }

  /// Number of time steps
  fn n(&self) -> usize;

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize>;

  /// Malliavin derivative of the process
  #[cfg(feature = "malliavin")]
  fn malliavin(&self) -> Array1<T> {
    unimplemented!()
  }
}

/// Trait for 2D sampling of stochastic processes
pub trait Sampling2DExt<T: Clone + Send + Sync + Zero>: Send + Sync {
  /// Sample the process
  fn sample(&self) -> [Array1<T>; 2];

  /// Parallel sampling
  fn sample_par(&self) -> [Array2<T>; 2] {
    if self.m().is_none() {
      panic!("m must be specified for parallel sampling");
    }

    let m = self.m().unwrap();
    let xs1 = Arc::new(Mutex::new(Array2::zeros((self.m().unwrap(), self.n()))));
    let xs2 = Arc::new(Mutex::new(Array2::zeros((self.m().unwrap(), self.n()))));

    (0..m).into_par_iter().for_each(|i| {
      let [x1, x2] = self.sample();
      xs1.lock().unwrap().row_mut(i).assign(&x1);
      xs2.lock().unwrap().row_mut(i).assign(&x2);
    });

    let xs1 = xs1.lock().unwrap().clone();
    let xs2 = xs2.lock().unwrap().clone();
    [xs1, xs2]
  }

  /// Number of time steps
  fn n(&self) -> usize;

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize>;

  /// Malliavin derivative of the process
  #[cfg(feature = "malliavin")]
  fn malliavin(&self) -> [Array1<T>; 2] {
    unimplemented!()
  }

  /// Set CUDA support
  #[cfg(feature = "cuda")]
  fn set_cuda(&mut self, cuda: bool) {}
}

/// Trait for 3D sampling of stochastic processes
pub trait Sampling3DExt<T: Clone + Send + Sync + Zero>: Send + Sync {
  /// Sample the process
  fn sample(&self) -> [Array1<T>; 3];

  /// Parallel sampling
  fn sample_par(&self) -> [Array2<T>; 3] {
    unimplemented!()
  }

  /// Number of time steps
  fn n(&self) -> usize;

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize>;
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
