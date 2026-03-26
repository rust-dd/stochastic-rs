//! # Yield Curve Construction
//!
//! Bootstrapping, parametric models, and multi-curve framework for interest rate term structures.
//!
//! Reference: Hagan & West, "Methods for Constructing a Yield Curve", Wilmott Magazine (2006).
//!
//! Reference: Nelson & Siegel, "Parsimonious Modeling of Yield Curves",
//! Journal of Business, 60(4), 473-489 (1987).
//!
//! Reference: Svensson, "Estimating and Interpreting Forward Interest Rates: Sweden 1992-1994",
//! IMF Working Paper 94/114 (1994).
//!
//! Reference: Bianchetti, "Two Curves, One Price", arXiv:0905.2770 (2009).
//!
//! $$
//! D(t) = e^{-\int_0^t f(s)\,ds}, \quad r(t) = -\frac{\ln D(t)}{t}, \quad f(t_1, t_2) = -\frac{\ln D(t_2) - \ln D(t_1)}{t_2 - t_1}
//! $$

pub mod bootstrap;
pub mod discount_curve;
pub mod interpolation;
#[cfg(feature = "openblas")]
pub(crate) mod linalg;
pub mod multi_curve;
pub mod nelson_siegel;
#[cfg(feature = "openblas")]
pub mod svensson;
pub mod types;

pub use bootstrap::{bootstrap, bootstrap_iterative};
pub use discount_curve::DiscountCurve;
pub use interpolation::interpolate_discount_factor;
pub use multi_curve::MultiCurve;
pub use nelson_siegel::NelsonSiegel;
#[cfg(feature = "openblas")]
pub use svensson::Svensson;
pub use types::{Compounding, CurvePoint, Instrument, InterpolationMethod};
