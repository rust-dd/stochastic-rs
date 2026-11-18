//! Extreme-value theory: the Hill tail index, GPD peaks over threshold and GEV block maxima.
//!
//! $$
//! \hat\xi^{H}_{k} = \frac1k\sum_{i=1}^{k}\log X_{(i)} - \log X_{(k+1)},\qquad
//! \widehat{\mathrm{VaR}}_p = u + \frac{\hat\sigma}{\hat\xi}
//! \Bigl[\Bigl(\frac{1-p}{n_u/n}\Bigr)^{-\hat\xi} - 1\Bigr],\qquad
//! z_m = \hat\mu - \frac{\hat\sigma}{\hat\xi}\Bigl[1 - \bigl(-\log(1 - 1/m)\bigr)^{-\hat\xi}\Bigr].
//! $$
//!
//! Reference: Hill (1975), Annals of Statistics 3(5), 1163-1174, DOI: 10.1214/aos/1176343247
//! Reference: Pickands (1975), Annals of Statistics 3(1), 119-131, DOI: 10.1214/aos/1176343003
//! Reference: Balkema & de Haan (1974), Ann. Probab. 2(5), 792-804, DOI: 10.1214/aop/1176996548
//! Reference: Hosking & Wallis (1987), Technometrics 29(3), 339-349, 10.1080/00401706.1987.10488243
//! Reference: Coles (2001), Springer Series in Statistics, ch. 3-4, DOI: 10.1007/978-1-4471-3675-0
//! Reference: McNeil, Frey & Embrechts (2015), Quantitative Risk Management, 2nd ed., §5.2

pub mod gev;
pub mod gpd;
pub mod hill;
#[cfg(test)]
mod tests;

pub use gev::GevFit;
pub use gev::block_maxima;
pub use gev::fit_gev;
pub use gpd::GpdFit;
pub use gpd::GpdPwmFit;
pub use gpd::PotFit;
pub use gpd::fit_gpd;
pub use gpd::fit_gpd_pwm;
pub use gpd::fit_pot;
pub use gpd::mean_excess;
pub use hill::HillFit;
pub use hill::fit_hill;
