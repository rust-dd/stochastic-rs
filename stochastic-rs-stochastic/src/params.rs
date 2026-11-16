//! Constructor vocabulary shared by several processes.

/// How a square-root state is kept non-negative after an Euler step: floored at zero or reflected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum VarianceBoundary {
  /// `max(x, 0)`.
  #[default]
  Floor,
  /// `|x|`.
  Reflect,
}

/// No process carries a `use_sym` flag: the step's boundary is a [`VarianceBoundary`].
///
/// ```compile_fail,E0560
/// use stochastic_rs_stochastic::diffusion::cir::Cir;
/// let c = Cir::<f64> { use_sym: Some(true), ..Cir::default() };
/// ```
#[cfg(doctest)]
pub struct UseSymIsGone;
