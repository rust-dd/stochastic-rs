//! Realized volatility and microstructure-noise estimators from intraday log-returns, and HAR-RV.
//!
//! $$
//! RV_t=\sum_{i=1}^{n} r_{t,i}^2,\qquad
//! BV_t=\frac{\pi}{2}\sum_{i=2}^{n} |r_{t,i-1}|\,|r_{t,i}|,\qquad
//! K(X)=\gamma_0(X)+\sum_{h=1}^{H} k\!\left(\tfrac{h-1}{H}\right)
//! \bigl(\gamma_h(X)+\gamma_{-h}(X)\bigr).
//! $$
//!
//! Reference: Andersen & Bollerslev (1998), Int. Econ. Rev. 39(4), 885-905, DOI: 10.2307/2527343
//! Reference: Barndorff-Nielsen & Shephard (2004), JFEC 2(1), 1-37, DOI: 10.1093/jjfinec/nbh001
//! Reference: Barndorff-Nielsen et al. (2008), Econometrica 76(6), 1481-1536, DOI: 10.3982/ECTA6495
//! Reference: Zhang et al. (2005), JASA 100(472), 1394-1411, DOI: 10.1198/016214505000000169
//! Reference: Jacod et al. (2009), SPA 119(7), 2249-2276, DOI: 10.1016/j.spa.2008.11.004
//! Reference: Lee & Mykland (2008), Rev. Financ. Stud. 21(6), 2535-2563, DOI: 10.1093/rfs/hhm056
//! Reference: Corsi (2009), JFEC 7(2), 174-196, DOI: 10.1093/jjfinec/nbp001
//! Reference: Barndorff-Nielsen et al. (2010), Volatility & Time Series Econom., 117-136, [DOI][b]
//!
//! [b]: https://doi.org/10.1093/acprof:oso/9780199549498.003.0007

pub mod bipower;
pub mod ewma;
pub mod har;
pub mod kernel;
pub mod lee_mykland;
pub mod pre_averaging;
pub mod two_scale;
pub mod variance;

pub use bipower::BnsJumpTest;
pub use bipower::bipower_variation;
pub use bipower::bns_jump_test;
pub use bipower::medrv;
pub use bipower::minrv;
pub use bipower::tripower_quarticity;
pub use ewma::EwmaResult;
pub use ewma::RISKMETRICS_DAILY_LAMBDA;
pub use ewma::ewma_variance;
pub use ewma::riskmetrics_variance;
pub use har::HarFit;
pub use har::HarRv;
pub use kernel::KernelType;
pub use kernel::realized_kernel;
pub use lee_mykland::LEE_MYKLAND_C;
pub use lee_mykland::LeeMyklandTest;
pub use lee_mykland::lee_mykland_test;
pub use lee_mykland::lee_mykland_window;
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
