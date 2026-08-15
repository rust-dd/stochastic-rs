//! # Goodness-of-fit tests
//!
//! Tests whether a sample was drawn from a **fully specified** target
//! distribution (every parameter fixed in advance — never fitted from
//! the sample under test; see [`kolmogorov_smirnov`]'s module doc for
//! why that distinction changes which critical values apply).
//!
//! | Test | Use on | Notes |
//! |------|--------|-------|
//! | [`kolmogorov_smirnov::kolmogorov_smirnov_test`] | continuous samples | `D_n` vs. Kolmogorov (1933) / Smirnov (1948) / Massey (1951) critical values |
//! | [`chi_square::chi_square_gof_test`] | discrete / binned counts | Pearson (1900) `\chi^2`, Cochran (1954) bin-pooling ([`chi_square::pool_integer_bins`]) |
//!
//! ## Choosing between them
//!
//! KS's asymptotic theory assumes a continuous null CDF; on an integer
//! lattice the empirical CDF's jumps coincide with the null's own jumps
//! at every support point, so the same statistic is no longer
//! calibrated the same way. Bin the data and use
//! [`chi_square::chi_square_gof_test`] for any discrete distribution
//! (Binomial, Poisson, Geometric, Hypergeometric, Skellam, …); use
//! [`kolmogorov_smirnov::kolmogorov_smirnov_test`] directly for
//! continuous ones.
//!
//! ## Cross-module pairing with [`crate::distributions`]
//!
//! Both tests are written against an arbitrary `cdf` closure, so the
//! natural reference is a distribution's own
//! [`DistributionExt::cdf`](stochastic_rs_distributions::traits::DistributionExt::cdf) —
//! closing the loop with that trait's own samplers rather than
//! requiring a second, independent reference implementation:
//!
//! ```
//! use ndarray::ArrayView1;
//! use stochastic_rs_core::simd_rng::Deterministic;
//! use stochastic_rs_distributions::DistributionExt;
//! use stochastic_rs_distributions::gamma::SimdGamma;
//! use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::{
//!     KolmogorovSmirnovConfig, kolmogorov_smirnov_test,
//! };
//!
//! let dist = SimdGamma::<f64>::new(2.5, 1.5, &Deterministic::new(42));
//! let mut sample = vec![0.0; 20_000];
//! dist.fill_slice(&mut sample);
//!
//! let res = kolmogorov_smirnov_test(
//!     ArrayView1::from(&sample),
//!     |x| dist.cdf(x),
//!     KolmogorovSmirnovConfig::default(),
//! );
//! assert!(!res.reject, "KS should not reject a sampler against its own cdf");
//! ```

pub mod chi_square;
pub mod kolmogorov_smirnov;
