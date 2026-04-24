//! # Credit Models
//!
//! $$
//! Q(t)=\mathbb{P}(\tau>t)=\exp\!\left(-\int_0^t h(s)\,ds\right),\qquad
//! \mathrm{PV}_{\text{prot}}=N(1-R)\int_0^T D(u)\,dQ(u)
//! $$
//!
//! Structural and reduced-form credit models, CDS pricing under the ISDA-style
//! hazard-rate framework, hazard-curve bootstrapping from a CDS term structure,
//! and credit rating migration matrices.
//!
//! Reference: Merton, "On the Pricing of Corporate Debt: The Risk Structure of
//! Interest Rates", Journal of Finance, 29(2), 449–470 (1974).
//! DOI: 10.1111/j.1540-6261.1974.tb03058.x
//!
//! Reference: Jarrow, Lando & Turnbull, "A Markov Model for the Term Structure
//! of Credit Risk Spreads", Review of Financial Studies, 10(2), 481–523 (1997).
//! DOI: 10.1093/rfs/10.2.481
//!
//! Reference: O'Kane & Turnbull, "Valuation of Credit Default Swaps", Lehman
//! Brothers Quantitative Credit Research Quarterly (2003).
//!
//! Reference: Brigo & Mercurio, "Interest Rate Models — Theory and Practice",
//! Springer, 2nd ed. (2006), Chapter 22. DOI: 10.1007/978-3-540-34604-3
//!
//! Reference: Pfeuffer, dos Reis & Smith, "Capturing Model Risk and Rating
//! Momentum in the Estimation of Probabilities of Default and Credit Rating
//! Migrations", arXiv:1809.09889 (2018).

pub mod bootstrap;
pub mod cds;
pub mod merton;
pub mod migration;
pub mod survival_curve;

pub use bootstrap::CdsQuote;
pub use bootstrap::bootstrap_hazard;
pub use cds::CdsPosition;
pub use cds::CdsValuation;
pub use cds::CreditDefaultSwap;
pub use merton::MertonStructural;
pub use migration::GeneratorMatrix;
pub use migration::TransitionMatrix;
pub use survival_curve::HazardInterpolation;
pub use survival_curve::HazardRateCurve;
pub use survival_curve::SurvivalCurve;
pub use survival_curve::SurvivalPoint;
