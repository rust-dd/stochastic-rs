//! Cointegration tests and VECM estimation, Granger causality, hidden Markov models, changepoints.
//!
//! $$
//! \Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i} + \mu + \varepsilon_t,
//! \qquad
//! \mathrm{rank}(\Pi) = r \in \{0, 1, \ldots, K\}.
//! $$
//!
//! Reference: Engle & Granger (1987), Econometrica 55(2), 251-276, DOI: 10.2307/1913236
//! Reference: Johansen (1988), JEDC 12(2-3), 231-254, DOI: 10.1016/0165-1889(88)90041-3
//! Reference: MacKinnon, Haug & Michelis (1999), J. Appl. Econometrics 14(5), 563-577, [DOI][mhm]
//! Reference: Granger (1969), Econometrica 37(3), 424-438, DOI: 10.2307/1912791
//! Reference: Baum et al. (1970), Ann. Math. Statist. 41(1), 164-171, DOI: 10.1214/aoms/1177697196
//! Reference: Killick et al. (2012), JASA 107(500), 1590-1598, DOI: 10.1080/01621459.2012.737745
//! Reference: Page (1954), Biometrika 41(1/2), 100-115, DOI: 10.2307/2333009
//!
//! [mhm]: https://doi.org/10.1002/(SICI)1099-1255(199909/10)14:5<563::AID-JAE530>3.0.CO;2-R

pub mod changepoint;
pub mod cointegration;
pub mod granger;
pub mod hmm;

pub use changepoint::ChangepointCusumResult;
pub use changepoint::PeltResult;
pub use changepoint::cusum;
pub use changepoint::pelt;
pub use cointegration::EngleGrangerConfig;
pub use cointegration::EngleGrangerTest;
pub use cointegration::JohansenTest;
pub use cointegration::VecmFit;
pub use cointegration::engle_granger_test;
pub use cointegration::fit_vecm;
pub use cointegration::johansen_test;
pub use granger::GrangerTest;
pub use granger::granger_causality_test;
pub use hmm::GaussianHmm;
pub use hmm::HmmFit;
