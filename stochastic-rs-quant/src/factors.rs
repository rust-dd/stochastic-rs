//! Principal-component factors, Ledoit–Wolf shrinkage covariance, the two-pass Fama–MacBeth
//! regression and cointegrated pairs trading.
//!
//! $$
//! \hat\Sigma_{LW} = \alpha\,\bar s\,I + (1-\alpha)\,S,\qquad
//! \alpha = \min\!\bigl(1,\,b^2 / d^2\bigr).
//! $$
//!
//! Reference: Ledoit & Wolf (2004), J. Multivariate Anal. 88(2), DOI 10.1016/S0047-259X(03)00096-4
//! Reference: Ledoit & Wolf (2004), J. Portfolio Management 30(4), DOI 10.3905/jpm.2004.110
//! Reference: Fama & MacBeth (1973), J. Political Economy 81(3), 607–636, DOI 10.1086/260061
//! Reference: Engle & Granger (1987), Econometrica 55(2), 251–276, DOI 10.2307/1913236
//! Reference: Gatev, Goetzmann & Rouwenhorst (2006), Rev. Financ. Stud. 19(3), 10.1093/rfs/hhj020

pub mod fama_macbeth;
pub mod pairs;
pub mod pca;
pub mod shrinkage;

pub use fama_macbeth::FamaMacBethResult;
pub use fama_macbeth::fama_macbeth;
pub use fama_macbeth::try_fama_macbeth;
pub use pairs::PairsSignal;
pub use pairs::PairsStrategy;
pub use pairs::pairs_signals;
pub use pairs::try_pairs_signals;
pub use pca::PcaResult;
pub use pca::pca_decompose;
pub use pca::try_pca_decompose;
pub use shrinkage::ledoit_wolf_shrinkage;
pub use shrinkage::sample_covariance;
