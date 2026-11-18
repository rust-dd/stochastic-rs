//! Optimal execution under price impact, Kyle's adverse-selection equilibria, propagator
//! impact and spread estimators. The $\kappa$ below is discrete-time Almgren–Chriss.
//!
//! $$
//! x_k = X\,\frac{\sinh(\kappa(T-t_k))}{\sinh(\kappa T)},\qquad
//! \kappa = \frac{1}{\tau}\,\mathrm{arccosh}\!\left(
//!   1 + \frac{\lambda\sigma^2 \tau^2}{2\tilde\eta}\right),
//! \quad \tilde\eta = \eta - \gamma\tau/2.
//! $$
//!
//! Reference: Almgren & Chriss (2001), J. Risk 3(2), 5–39, DOI 10.21314/JOR.2001.041
//! Reference: Kyle (1985), Econometrica 53(6), 1315–1335, DOI 10.2307/1913210
//! Reference: Bouchaud, Gefen, Potters & Wyart (2004), Quant. Finance 4(2), 176–190
//! Reference: Obizhaeva & Wang (2013), J. Financial Markets 16(1), DOI 10.1016/j.finmar.2012.09.001
//! Reference: Roll (1984), Journal of Finance 39(4), DOI 10.1111/j.1540-6261.1984.tb03897.x
//! Reference: Hasbrouck (1991), Journal of Finance 46(1), DOI 10.1111/j.1540-6261.1991.tb03749.x
//! Reference: Corwin & Schultz (2012), J. Finance 67(2), DOI 10.1111/j.1540-6261.2012.01729.x

pub mod almgren_chriss;
pub mod impact;
pub mod kyle;
pub mod spread;

pub use almgren_chriss::AlmgrenChrissParams;
pub use almgren_chriss::AlmgrenChrissPlan;
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
