//! # Econometrics
//!
//! Cointegration tests, Granger causality, hidden Markov regime models and
//! changepoint detection on time series.
//!
//! $$
//! \Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i} + \mu + \varepsilon_t,
//! \qquad
//! \mathrm{rank}(\Pi) = r \in \{0, 1, \ldots, K\}.
//! $$
//!
//! Most tests require the `openblas` feature for matrix decompositions.
//!
//! # References
//! - Engle, Granger, "Co-Integration and Error Correction: Representation,
//!   Estimation, and Testing", Econometrica, 55(2), 251-276 (1987).
//!   DOI: 10.2307/1913236
//! - Johansen, "Statistical Analysis of Cointegration Vectors", Journal of
//!   Economic Dynamics and Control, 12(2-3), 231-254 (1988).
//!   DOI: 10.1016/0165-1889(88)90041-3
//! - Granger, "Investigating Causal Relations by Econometric Models and
//!   Cross-Spectral Methods", Econometrica, 37(3), 424-438 (1969).
//!   DOI: 10.2307/1912791
//! - Baum, Petrie, Soules, Weiss, "A Maximization Technique Occurring in the
//!   Statistical Analysis of Probabilistic Functions of Markov Chains",
//!   Annals of Mathematical Statistics, 41(1), 164-171 (1970).
//!   DOI: 10.1214/aoms/1177697196
//! - Killick, Fearnhead, Eckley, "Optimal Detection of Changepoints With a
//!   Linear Computational Cost", Journal of the American Statistical
//!   Association, 107(500), 1590-1598 (2012). DOI: 10.1080/01621459.2012.737745
//! - Page, "Continuous Inspection Schemes", Biometrika, 41(1/2), 100-115
//!   (1954). DOI: 10.2307/2333009

pub mod changepoint;
#[cfg(feature = "openblas")]
pub mod cointegration;
#[cfg(feature = "openblas")]
pub mod granger;
#[cfg(feature = "openblas")]
pub mod hmm;

pub use changepoint::CusumResult;
pub use changepoint::PeltResult;
pub use changepoint::cusum;
pub use changepoint::pelt;
#[cfg(feature = "openblas")]
pub use cointegration::EngleGrangerResult;
#[cfg(feature = "openblas")]
pub use cointegration::JohansenResult;
#[cfg(feature = "openblas")]
pub use cointegration::engle_granger_test;
#[cfg(feature = "openblas")]
pub use cointegration::johansen_test;
#[cfg(feature = "openblas")]
pub use granger::GrangerResult;
#[cfg(feature = "openblas")]
pub use granger::granger_causality;
#[cfg(feature = "openblas")]
pub use hmm::GaussianHmm;
#[cfg(feature = "openblas")]
pub use hmm::HmmFit;
