//! Bootstrapping, parametric models and the multi-curve framework for interest-rate term
//! structures.
//!
//! $$
//! D(t) = e^{-\int_0^t f(s)\,ds}, \quad r(t) = -\frac{\ln D(t)}{t}, \quad
//! f(t_1, t_2) = -\frac{\ln D(t_2) - \ln D(t_1)}{t_2 - t_1}
//! $$
//!
//! Reference: Hagan & West (2006), "Methods for Constructing a Yield Curve", Wilmott Magazine
//! Reference: Nelson & Siegel (1987), Journal of Business 60(4), 473–489
//! Reference: Svensson (1994), "Estimating and Interpreting Forward Interest Rates", IMF WP 94/114
//! Reference: Bianchetti (2009), "Two Curves, One Price", arXiv:0905.2770
//! Reference: Ametrano & Bianchetti (2013), SSRN 2219548

pub mod bootstrap;
pub mod discount_curve;
mod dual_curve;
pub mod interpolation;
pub mod multi_curve;
pub mod nelson_siegel;
pub mod svensson;
pub mod types;

pub use bootstrap::BootstrapSpec;
pub use bootstrap::bootstrap;
pub use discount_curve::DiscountCurve;
pub use interpolation::interpolate_discount_factor;
pub use multi_curve::MultiCurve;
pub use nelson_siegel::NelsonSiegel;
pub use svensson::Svensson;
pub use types::BootstrapInstrument;
pub use types::Compounding;
pub use types::CurvePoint;
pub use types::InterpolationMethod;
