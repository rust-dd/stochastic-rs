//! Normality tests: Jarque–Bera on moments, Anderson–Darling on tails, Shapiro–Francia on shape.
//!
//! [`stochastic_rs_distributions::normal::SimdNormal`] generates the sample these tests check:
//!
//! ```
//! use ndarray::ArrayView1;
//! use stochastic_rs_core::simd_rng::Deterministic;
//! use stochastic_rs_distributions::DistributionSampler;
//! use stochastic_rs_distributions::SimdDistribution;
//! use stochastic_rs_distributions::normal::SimdNormal;
//! use stochastic_rs_stats::normality::jarque_bera::{
//!     JarqueBeraConfig, jarque_bera_test,
//! };
//!
//! // A fixed seed keeps this example deterministic — an `Unseeded` draw
//! // would make the assertion below a (tiny) source of doctest flakiness.
//! let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(42));
//! let mut sample = vec![0.0; 5_000];
//! dist.fill_slice(&mut sample);
//!
//! let res = jarque_bera_test(ArrayView1::from(&sample), JarqueBeraConfig::default())
//!     .expect("5 000 finite draws");
//! assert!(!res.rejects_null, "JB should not reject a true normal sample");
//! ```

pub mod anderson_darling;
pub mod jarque_bera;
pub mod shapiro_francia;
