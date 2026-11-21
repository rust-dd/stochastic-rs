//! Univariate margins that map raw series onto `[0, 1]` for a copula fit.
//!
//! $$
//! F_{X_1,\dots,X_d}(x)=C\left(F_1(x_1),\dots,F_d(x_d)\right)
//! $$
//!
pub mod gaussian;
