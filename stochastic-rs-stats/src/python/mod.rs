//! PyO3 wrappers for `stochastic-rs-stats`.
//!
//! Hypothesis tests (Jarque-Bera, Anderson-Darling, Shapiro-Francia, ADF, KPSS),
//! Hurst-exponent estimators (Fukasawa, fOU v1/v2), Heston MLE, and the realised
//! variance / bipower-variation / jump tests, all exposed as `#[pyclass]` types
//! that take a numpy array and return a result object with the test statistic
//! and either a p-value or boolean rejection flag.

#![cfg(feature = "python")]
#![allow(clippy::too_many_arguments)]

mod changepoint;
mod cointegration;
mod hmm;
mod hurst;
mod mcmc;
mod misc;
mod mle;
mod normality;
mod realized;
mod stationarity;

pub use changepoint::PyCusum;
pub use changepoint::PyPelt;
pub use changepoint::PyPeriodogramFFT;
#[cfg(feature = "openblas")]
pub use cointegration::PyEngleGranger;
#[cfg(feature = "openblas")]
pub use cointegration::PyGranger;
#[cfg(feature = "openblas")]
pub use cointegration::PyJohansen;
#[cfg(feature = "openblas")]
pub use hmm::PyGaussianHmm;
pub use hurst::PyDfa;
pub use hurst::PyFdResult;
pub use hurst::PyFouEstimate;
pub use hurst::PyFukasawaHurst;
pub use hurst::PyGph;
pub use hurst::PyHiguchi;
pub use hurst::PyHurstResult;
pub use hurst::PyRescaledRange;
pub use hurst::PyVariations;
pub use hurst::PyVariogram;
pub use hurst::PyWavelet;
pub use hurst::PyWhittle;
pub use mcmc::random_walk_metropolis;
pub use misc::PyGaussianKDE;
pub use misc::PyLeverage;
pub use misc::PyTailIndex;
pub use mle::PyHestonMLE;
pub use mle::PyHestonNMLECEKF;
pub use normality::PyAndersonDarling;
pub use normality::PyJarqueBera;
pub use normality::PyShapiroFrancia;
pub use realized::PyBNSJumpTest;
pub use realized::PyBipowerVariation;
#[cfg(feature = "openblas")]
pub use realized::PyHarRv;
pub use realized::PyPreAveragedVariance;
pub use realized::PyRealizedKernel;
pub use realized::PyRealizedMoments;
pub use realized::PyTwoScaleRV;
#[cfg(feature = "openblas")]
pub use stationarity::PyADFTest;
#[cfg(feature = "openblas")]
pub use stationarity::PyERSTest;
#[cfg(feature = "openblas")]
pub use stationarity::PyKPSSTest;
#[cfg(feature = "openblas")]
pub use stationarity::PyLeybourneMcCabeTest;
#[cfg(feature = "openblas")]
pub use stationarity::PyPhillipsPerronTest;
