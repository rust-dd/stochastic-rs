//! Structural and reduced-form credit models, CDS pricing on a hazard curve, hazard bootstrapping
//! from a CDS term structure, and rating migration matrices.
//!
//! $$
//! Q(t)=\mathbb{P}(\tau>t)=\exp\!\left(-\int_0^t h(s)\,ds\right),\qquad
//! \mathrm{PV}_{\text{prot}}=N(1-R)\int_0^T D(u)\,dQ(u)
//! $$
//!
//! Reference: Merton (1974), Journal of Finance 29(2), DOI 10.1111/j.1540-6261.1974.tb03058.x
//! Reference: Jarrow, Lando & Turnbull (1997), Rev. Financ. Stud. 10(2), DOI 10.1093/rfs/10.2.481
//! Reference: O'Kane & Turnbull (2003), "Valuation of Credit Default Swaps", Lehman Brothers QCR
//! Reference: Brigo & Mercurio (2006), 2nd ed., Ch. 22, DOI 10.1007/978-3-540-34604-3
//! Reference: Pfeuffer, dos Reis & Smith (2018), arXiv:1809.09889

pub mod bootstrap;
pub mod cds;
pub mod index;
pub mod merton;
pub mod migration;
pub mod survival_curve;
pub mod tranche;

pub use bootstrap::CdsQuote;
pub use bootstrap::bootstrap_hazard;
pub use cds::CdsPosition;
pub use cds::CdsValuation;
pub use cds::CreditDefaultSwap;
pub use index::CdsIndex;
pub use index::IndexName;
pub use index::IndexValuation;
pub use merton::MertonStructural;
pub use migration::GeneratorMatrix;
pub use migration::TransitionMatrix;
pub use survival_curve::HazardInterpolation;
pub use survival_curve::HazardRateCurve;
pub use survival_curve::SurvivalCurve;
pub use survival_curve::SurvivalPoint;
pub use tranche::CdoTranche;
pub use tranche::PoolName;
pub use tranche::TrancheValuation;
