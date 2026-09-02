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
mod distfit;
mod evt;
mod garch;
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
pub use cointegration::PyEngleGranger;
pub use cointegration::PyGranger;
pub use cointegration::PyJohansen;
pub use cointegration::PyVecm;
pub use distfit::PyGpdPwm;
pub use distfit::PyJohnsonSuFit;
pub use distfit::PySkewTFit;
pub use distfit::PyVarianceGammaFit;
pub use evt::PyGevFit;
pub use evt::PyGpdFit;
pub use evt::PyHillEstimator;
pub use evt::PyPotFit;
pub use evt::block_maxima;
pub use evt::mean_excess;
pub use garch::PyGarchFit;
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
pub use realized::PyEwmaVariance;
pub use realized::PyHarRv;
pub use realized::PyLeeMyklandJumpTest;
pub use realized::PyPreAveragedVariance;
pub use realized::PyRealizedKernel;
pub use realized::PyRealizedMoments;
pub use realized::PyTwoScaleRV;
pub use stationarity::PyADFTest;
pub use stationarity::PyERSTest;
pub use stationarity::PyKPSSTest;
pub use stationarity::PyLeybourneMcCabeTest;
pub use stationarity::PyPhillipsPerronTest;
