//! Interest-rate options — caps, floors, collars, and swaptions.
//!
//! $$
//! \mathrm{Caplet}=N\,\alpha\,P(t,t_p)\,[F\Phi(d_1)-K\Phi(d_2)],\quad
//! V_{\mathrm{swpn}}=A\,[s\Phi(d_1)-K\Phi(d_2)]
//! $$
//!
//! Closed-form Black-76, Bachelier, and SABR caplet/floorlet formulae with
//! European cap, floor, collar, and swaption wrappers on top of the library
//! curve and swap stack; Bermudan swaptions priced on the Hull-White trinomial
//! lattice from `quant::lattice`.
//!
//! Reference: P. S. Hagan et al., "Managing Smile Risk", Wilmott (2002).
//!
//! Reference: D. Brigo & F. Mercurio, "Interest Rate Models — Theory and
//! Practice", Springer 2nd ed. (2006), §1.5, §6.7, §13.
//!
//! Reference: J. Hull, "Options, Futures, and Other Derivatives", 11th ed.
//! (2021), §29.

pub mod bermudan;
pub mod cap;
pub mod caplet;
pub mod swaption;
pub mod types;
pub mod volatility;

pub use bermudan::BermudanSwaption;
pub use cap::Cap;
pub use cap::Collar;
pub use cap::Floor;
pub use caplet::bachelier_forward_caplet;
pub use caplet::bachelier_forward_floorlet;
pub use caplet::black_forward_caplet;
pub use caplet::black_forward_floorlet;
pub use caplet::caplet_price;
pub use swaption::EuropeanSwaption;
pub use types::BermudanSwaptionValuation;
pub use types::CapFloorValuation;
pub use types::CollarValuation;
pub use types::ExerciseDate;
pub use types::ExerciseSchedule;
pub use types::InterestRateOptionKind;
pub use types::SwaptionDirection;
pub use types::SwaptionValuation;
pub use types::TreeCouponSchedule;
pub use types::VolatilityQuoteKind;
pub use volatility::BachelierVolatility;
pub use volatility::BlackVolatility;
pub use volatility::SabrVolatility;
pub use volatility::VolatilityModel;
