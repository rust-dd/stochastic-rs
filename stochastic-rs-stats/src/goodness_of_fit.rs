//! Goodness-of-fit tests against a fully specified law: Kolmogorov–Smirnov and Pearson chi-square.
//!
//! KS needs a continuous CDF, so bin lattice data for chi-square; fit no parameter from the sample.
//!
//! Reference: Kolmogorov (1933), Giornale dell'Istituto Italiano degli Attuari 4, 83-91
//! Reference: Smirnov (1948), Annals of Mathematical Statistics 19(2), 279-281
//! Reference: Massey (1951), JASA 46(253), 68-78
//! Reference: Pearson (1900), Philosophical Magazine Series 5, 50(302), 157-175
//! Reference: Cochran (1954), Biometrics 10(4), 417-451
//!
//! ```
//! use ndarray::ArrayView1;
//! use stochastic_rs_core::simd_rng::Deterministic;
//! use stochastic_rs_distributions::DistributionExt;
//! use stochastic_rs_distributions::DistributionSampler;
//! use stochastic_rs_distributions::SimdDistribution;
//! use stochastic_rs_distributions::gamma::SimdGamma;
//! use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::{
//!     KolmogorovSmirnovConfig, kolmogorov_smirnov_test,
//! };
//!
//! let dist = SimdGamma::<f64>::new(2.5, 1.5);
//! let mut sample = vec![0.0; 20_000];
//! dist.seeded(&Deterministic::new(42)).fill_slice(&mut sample);
//!
//! let res = kolmogorov_smirnov_test(
//!     ArrayView1::from(&sample),
//!     |x| dist.cdf(x).unwrap(),
//!     KolmogorovSmirnovConfig::default(),
//! )
//! .expect("20 000 finite draws");
//! assert!(!res.rejects_null, "KS should not reject a sampler against its own cdf");
//! ```

pub mod chi_square;
pub mod kolmogorov_smirnov;
