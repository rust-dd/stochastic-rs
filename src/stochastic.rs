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

use std::cell::RefCell;

use ndarray::Array1;
use num_complex::Complex;

use crate::distributions::normal::SimdNormal;
pub use crate::traits::DistributionExt;
use crate::traits::FloatExt;
pub use crate::traits::ProcessExt;
pub use crate::traits::SimdFloatExt;

/// Default number of time steps
pub const N: usize = 1000;
/// Default initial value
pub const X0: f64 = 0.5;
/// Default spot price for financial models
pub const S0: f64 = 100.0;
/// Default strike price
pub const K: f64 = 100.0;

thread_local! {
  static STANDARD_NORMAL_F64: RefCell<Option<Box<SimdNormal<f64, 64>>>> = const { RefCell::new(None) };
  static STANDARD_NORMAL_F32: RefCell<Option<Box<SimdNormal<f32, 64>>>> = const { RefCell::new(None) };
  static FGN_SCRATCH_F64: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
  static FGN_SCRATCH_F32: RefCell<Vec<Complex<f32>>> = const { RefCell::new(Vec::new()) };
}

impl FloatExt for f64 {
  fn from_usize_(n: usize) -> Self {
    n as f64
  }

  fn fill_standard_normal_slice(out: &mut [Self]) {
    if out.is_empty() {
      return;
    }
    STANDARD_NORMAL_F64.with(|cell| {
      let mut slot = cell.borrow_mut();
      let dist = slot.get_or_insert_with(|| Box::new(SimdNormal::new(0.0, 1.0)));
      dist.fill_standard_fast(out);
    });
  }

  fn with_fgn_complex_scratch<R, F: FnOnce(&mut [Complex<Self>]) -> R>(len: usize, f: F) -> R {
    FGN_SCRATCH_F64.with(|scratch| {
      let mut scratch = scratch.borrow_mut();
      if scratch.len() < len {
        scratch.resize(len, Complex::new(0.0, 0.0));
      }
      f(&mut scratch[..len])
    })
  }

  fn normal_array(n: usize, mean: Self, std_dev: Self) -> Array1<Self> {
    assert!(std_dev > 0.0, "std_dev must be positive");
    let mut out = Array1::<f64>::zeros(n);
    if n == 0 {
      return out;
    }
    let out_slice = out
      .as_slice_mut()
      .expect("normal_array output must be contiguous");
    Self::fill_standard_normal_slice(out_slice);
    for x in out_slice.iter_mut() {
      *x = mean + std_dev * *x;
    }
    out
  }
}

impl FloatExt for f32 {
  fn from_usize_(n: usize) -> Self {
    n as f32
  }

  fn fill_standard_normal_slice(out: &mut [Self]) {
    if out.is_empty() {
      return;
    }
    STANDARD_NORMAL_F32.with(|cell| {
      let mut slot = cell.borrow_mut();
      let dist = slot.get_or_insert_with(|| Box::new(SimdNormal::new(0.0, 1.0)));
      dist.fill_standard_fast(out);
    });
  }

  fn with_fgn_complex_scratch<R, F: FnOnce(&mut [Complex<Self>]) -> R>(len: usize, f: F) -> R {
    FGN_SCRATCH_F32.with(|scratch| {
      let mut scratch = scratch.borrow_mut();
      if scratch.len() < len {
        scratch.resize(len, Complex::new(0.0, 0.0));
      }
      f(&mut scratch[..len])
    })
  }

  fn normal_array(n: usize, mean: Self, std_dev: Self) -> Array1<Self> {
    assert!(std_dev > 0.0, "std_dev must be positive");
    let mut out = Array1::<f32>::zeros(n);
    if n == 0 {
      return out;
    }
    let out_slice = out
      .as_slice_mut()
      .expect("normal_array output must be contiguous");
    Self::fill_standard_normal_slice(out_slice);
    for x in out_slice.iter_mut() {
      *x = mean + std_dev * *x;
    }
    out
  }
}
