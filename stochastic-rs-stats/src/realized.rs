//! # Realized Volatility & Microstructure-Noise Estimators
//!
//! High-frequency, model-free variance, jump and tail estimators built from
//! intraday log-returns plus the heterogeneous-autoregressive realized-volatility
//! forecasting model (HAR-RV).
//!
//! $$
//! RV_t=\sum_{i=1}^{n} r_{t,i}^2,\qquad
//! BV_t=\frac{\pi}{2}\sum_{i=2}^{n} |r_{t,i-1}|\,|r_{t,i}|,\qquad
//! K(X)=\gamma_0(X)+\sum_{h=1}^{H} k\!\left(\tfrac{h-1}{H}\right)\bigl(\gamma_h(X)+\gamma_{-h}(X)\bigr).
//! $$
//!
//! # References
//! - Andersen, Bollerslev, "Answering the Skeptics: Yes, Standard Volatility
//!   Models Do Provide Accurate Forecasts", International Economic Review,
//!   39(4), 885-905 (1998). DOI: 10.2307/2527343
//! - Barndorff-Nielsen, Shephard, "Power and Bipower Variation with Stochastic
//!   Volatility and Jumps", Journal of Financial Econometrics, 2(1), 1-37 (2004).
//!   DOI: 10.1093/jjfinec/nbh001
//! - Barndorff-Nielsen, Hansen, Lunde, Shephard, "Designing Realised Kernels to
//!   Measure the Ex-Post Variation of Equity Prices in the Presence of Noise",
//!   Econometrica, 76(6), 1481-1536 (2008). DOI: 10.3982/ECTA6495
//! - Zhang, Mykland, Aït-Sahalia, "A Tale of Two Time Scales: Determining
//!   Integrated Volatility With Noisy High-Frequency Data", Journal of the
//!   American Statistical Association, 100(472), 1394-1411 (2005).
//!   DOI: 10.1198/016214505000000169
//! - Jacod, Li, Mykland, Podolskij, Vetter, "Microstructure Noise in the
//!   Continuous Case: The Pre-Averaging Approach", Stochastic Processes and
//!   their Applications, 119(7), 2249-2276 (2009).
//!   DOI: 10.1016/j.spa.2008.11.004
//! - Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility",
//!   Journal of Financial Econometrics, 7(2), 174-196 (2009).
//!   DOI: 10.1093/jjfinec/nbp001
//! - Barndorff-Nielsen, Kinnebrock, Shephard, "Measuring Downside Risk —
//!   Realised Semivariance", in *Volatility and Time Series Econometrics:
//!   Essays in Honour of Robert F. Engle*, Oxford U. Press (2010).
//!   DOI: 10.1093/acprof:oso/9780199549498.003.0007

pub mod bipower;
#[cfg(feature = "openblas")]
pub mod har;
pub mod kernel;
pub mod pre_averaging;
pub mod two_scale;
pub mod variance;

pub use bipower::BnsJumpTest;
pub use bipower::bipower_variation;
pub use bipower::bns_jump_test;
pub use bipower::medrv;
pub use bipower::minrv;
pub use bipower::tripower_quarticity;
#[cfg(feature = "openblas")]
pub use har::HarFit;
#[cfg(feature = "openblas")]
pub use har::HarRv;
#[cfg(feature = "openblas")]
pub use har::har_features;
pub use kernel::KernelType;
pub use kernel::realized_kernel;
pub use pre_averaging::pre_averaged_variance;
pub use two_scale::multi_scale_rv;
pub use two_scale::two_scale_rv;
pub use variance::log_returns;
pub use variance::realized_kurtosis;
pub use variance::realized_quarticity;
pub use variance::realized_semivariance;
pub use variance::realized_skewness;
pub use variance::realized_variance;
pub use variance::realized_volatility;
