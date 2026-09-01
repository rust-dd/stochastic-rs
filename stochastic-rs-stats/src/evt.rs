//! # Extreme-value theory
//!
//! Tail-index and tail-risk estimation from the two classical EVT limit
//! results — the Hill estimator of the tail index, the generalised-Pareto
//! peaks-over-threshold model with its Value-at-Risk and expected-shortfall
//! formulas, and the GEV block-maxima model with return levels:
//!
//! $$
//! \hat\xi^{H}_{k} = \frac1k\sum_{i=1}^{k}\log X_{(i)} - \log X_{(k+1)},\qquad
//! \widehat{\mathrm{VaR}}_p = u + \frac{\hat\sigma}{\hat\xi}\Bigl[\Bigl(\frac{1-p}{n_u/n}\Bigr)^{-\hat\xi} - 1\Bigr],\qquad
//! z_m = \hat\mu - \frac{\hat\sigma}{\hat\xi}\Bigl[1 - \bigl(-\log(1 - 1/m)\bigr)^{-\hat\xi}\Bigr].
//! $$
//!
//! # References
//! - Hill, "A Simple General Approach to Inference About the Tail of a
//!   Distribution", Annals of Statistics, 3(5), 1163-1174 (1975).
//!   DOI: 10.1214/aos/1176343247
//! - Pickands, "Statistical Inference Using Extreme Order Statistics",
//!   Annals of Statistics, 3(1), 119-131 (1975). DOI: 10.1214/aos/1176343003
//! - Balkema, de Haan, "Residual Life Time at Great Age", Annals of
//!   Probability, 2(5), 792-804 (1974). DOI: 10.1214/aop/1176996548
//! - Coles, *An Introduction to Statistical Modeling of Extreme Values*,
//!   Springer (2001), ch. 3-4. DOI: 10.1007/978-1-4471-3675-0
//! - McNeil, Frey, Embrechts, *Quantitative Risk Management*, 2nd ed.,
//!   Princeton University Press (2015), §5.2.

pub mod gev_fit;
pub mod gpd_fit;
pub mod hill;
#[cfg(test)]
mod tests;

pub use gev_fit::GevFit;
pub use gev_fit::block_maxima;
pub use gev_fit::gev_fit;
pub use gpd_fit::GpdFit;
pub use gpd_fit::PotFit;
pub use gpd_fit::gpd_fit;
pub use gpd_fit::mean_excess;
pub use gpd_fit::pot_fit;
pub use hill::HillResult;
pub use hill::hill_estimator;
