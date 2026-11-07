//! Low-discrepancy point sets: Sobol on the full Joe–Kuo table with Owen-type scrambling, and Halton.

pub mod halton;
pub mod sobol;

pub use halton::HaltonSeq;
pub use sobol::MAX_DIMENSIONS;
pub use sobol::SobolSeq;
