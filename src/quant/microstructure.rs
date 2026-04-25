//! # Market Microstructure & Optimal Execution
//!
//! Optimal execution under permanent and temporary price impact, structural
//! single- and multi-period adverse-selection equilibria, transient
//! propagator-style impact, and microstructure-noise spread estimators.
//!
//! $$
//! x_k = X\,\frac{\sinh(\kappa(T-t_k))}{\sinh(\kappa T)},\qquad
//! \kappa = \sqrt{\lambda\sigma^2/\tilde\eta},\quad \tilde\eta = \eta - \gamma\tau/2.
//! $$
//!
//! # References
//! - Almgren, Chriss, "Optimal Execution of Portfolio Transactions", Journal
//!   of Risk, 3(2), 5-39 (2001). DOI: 10.21314/JOR.2001.041
//! - Kyle, "Continuous Auctions and Insider Trading", Econometrica, 53(6),
//!   1315-1335 (1985). DOI: 10.2307/1913210
//! - Bouchaud, Gefen, Potters, Wyart, "Fluctuations and Response in Financial
//!   Markets: The Subtle Nature of Random Price Changes", Quantitative
//!   Finance, 4(2), 176-190 (2004). DOI: 10.1080/14697680400000022
//! - Obizhaeva, Wang, "Optimal Trading Strategy and Supply/Demand Dynamics",
//!   Journal of Financial Markets, 16(1), 1-32 (2013).
//!   DOI: 10.1016/j.finmar.2012.09.001
//! - Roll, "A Simple Implicit Measure of the Effective Bid-Ask Spread in an
//!   Efficient Market", Journal of Finance, 39(4), 1127-1139 (1984).
//!   DOI: 10.1111/j.1540-6261.1984.tb03897.x
//! - Hasbrouck, "Measuring the Information Content of Stock Trades", Journal
//!   of Finance, 46(1), 179-207 (1991). DOI: 10.1111/j.1540-6261.1991.tb03749.x
//! - Corwin, Schultz, "A Simple Way to Estimate Bid-Ask Spreads from Daily
//!   High and Low Prices", Journal of Finance, 67(2), 719-760 (2012).
//!   DOI: 10.1111/j.1540-6261.2012.01729.x

pub mod almgren_chriss;
pub mod impact;
pub mod kyle;
pub mod spread;

pub use almgren_chriss::AlmgrenChrissParams;
pub use almgren_chriss::AlmgrenChrissPlan;
pub use almgren_chriss::ExecutionDirection;
pub use almgren_chriss::optimal_execution;
pub use impact::ImpactKernel;
pub use impact::propagator_impact_path;
pub use impact::propagator_price_impact;
pub use kyle::KyleEquilibrium;
pub use kyle::multi_period_kyle;
pub use kyle::single_period_kyle;
pub use spread::corwin_schultz_spread;
pub use spread::effective_spread;
pub use spread::roll_spread;
