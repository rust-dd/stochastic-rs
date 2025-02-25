//! # Stochastic Process Simulation Modules
//!
//! `stochastic` provides various modules to simulate and analyze stochastic processes efficiently.
//!
//! ## Modules
//!
//! | Module          | Description                                                                                                                                                                       |
//! |-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
//! | **diffusion**    | Handles diffusion processes, such as Brownian motion and Geometric Brownian motion, commonly used in physics and finance to model random behavior over time.                                                            |
//! | **interest**     | Provides models for simulating stochastic interest rates, including well-known models like the Cox-Ingersoll-Ross (CIR) model used in financial mathematics.                                                             |
//! | **jump**         | Implements jump processes, where sudden changes occur at random intervals, such as in the Poisson process or in financial models like the Bates model.                                                                  |
//! | **malliavin**    | Tools for working with the Malliavin calculus, which is used to compute derivatives of stochastic processes for sensitivity analysis and other advanced applications.                                                     |
//! | **noise**        | Generates various noise processes, including Gaussian and fractional Gaussian noise, which are essential for simulating random perturbations in stochastic models.                                                       |
//! | **process**      | Provides general abstractions and implementations for creating, simulating, and sampling stochastic processes, supporting both regular and parallelized workflows.                                                       |
//! | **volatility**   | Focuses on modeling stochastic volatility, including processes like the Heston model, which are used to simulate changes in volatility over time in financial markets.                                                    |
//!

pub mod autoregressive;
pub mod diffusion;
pub mod interest;
pub mod isonormal;
pub mod jump;
#[cfg(feature = "malliavin")]
pub mod malliavin;
pub mod noise;
pub mod process;
pub mod volatility;

use std::sync::{Arc, Mutex};

use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Axis};
use ndrustfft::Zero;
use num_complex::Complex64;

pub const N: usize = 1000;
pub const X0: f64 = 0.5;
pub const S0: f64 = 100.0;
pub const K: f64 = 100.0;

pub trait Sampling<T: Clone + Send + Sync + Zero>: Send + Sync {
  /// Sample the process
  fn sample(&self) -> Array1<T>;

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
}

pub trait SamplingVector<T: Clone + Send + Sync + Zero>: Send + Sync {
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

pub trait Sampling2D<T: Clone + Send + Sync + Zero>: Send + Sync {
  /// Sample the process
  fn sample(&self) -> [Array1<T>; 2];

  /// Parallel sampling
  fn sample_par(&self) -> [Array2<T>; 2] {
    if self.m().is_none() {
      panic!("m must be specified for parallel sampling");
    }

    let m = self.m().unwrap(); // m értékét előre kinyerjük, hogy ne kelljen többször unwrap-elni
    let xs1 = Arc::new(Mutex::new(Array2::zeros((self.m().unwrap(), self.n()))));
    let xs2 = Arc::new(Mutex::new(Array2::zeros((self.m().unwrap(), self.n()))));

    (0..m).into_par_iter().for_each(|i| {
      let [x1, x2] = self.sample(); // Minden szálon mintavételezünk
      xs1.lock().unwrap().row_mut(i).assign(&x1); // Az első mintavételezés eredményét beírjuk az első mátrix i. sorába
      xs2.lock().unwrap().row_mut(i).assign(&x2); // A második mintavételezés eredményét beírjuk a második mátrix i. sorába
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
}

pub trait Sampling3D<T: Clone + Send + Sync + Zero>: Send + Sync {
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

pub trait Distribution {
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
