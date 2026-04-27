//! # Normality tests
//!
//! Three classical normality tests for samples on the real line:
//!
//! | Test | Power on | Notes |
//! |------|----------|-------|
//! | [`jarque_bera::jarque_bera_test`] | skewness + excess kurtosis | $\chi^2_2$ asymptotic, very fast |
//! | [`anderson_darling::anderson_darling_normal_test`] | tail behaviour | Stephens p-value, hard on heavy tails |
//! | [`shapiro_francia::shapiro_francia_test`] | full shape | bootstrap p-value, slowest but powerful |
//!
//! ## Cross-module pairing with [`crate::distributions`]
//!
//! The [`stochastic_rs_distributions::normal::SimdNormal`] sampler from the
//! `distributions` crate is the canonical generator for normal samples; the
//! tests here are the canonical *checkers* for whether a sample looks
//! normal. End-to-end roundtrip:
//!
//! ```ignore
//! use ndarray::ArrayView1;
//! use stochastic_rs_distributions::normal::SimdNormal;
//! use stochastic_rs_stats::normality::jarque_bera::{
//!     JarqueBeraConfig, jarque_bera_test,
//! };
//!
//! let dist = SimdNormal::<f64>::new(0.0, 1.0);
//! let mut sample = vec![0.0; 5_000];
//! dist.fill_slice_fast(&mut sample);
//!
//! let res = jarque_bera_test(ArrayView1::from(&sample), JarqueBeraConfig::default());
//! assert!(!res.reject_normality, "JB should not reject a true normal sample");
//! ```
//!
//! Use the same pattern with `SimdLogNormal`, `SimdStudentT`, etc. to
//! quantify how strongly each test rejects departures from normality.

pub mod anderson_darling;
pub mod jarque_bera;
pub mod shapiro_francia;
