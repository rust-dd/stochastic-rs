//! # Truncated distributions
//!
//! Truncated $\mathrm{Normal}$, $\mathrm{Beta}$, $\mathrm{Gamma}$ and
//! $\mathrm{Exponential}$ distributions restricted to a closed interval
//! $[a, b]$ (with $-\infty \le a < b \le +\infty$ allowed per family).
//!
//! ## Sampling
//!
//! - **Truncated Normal:** plain rejection from the base
//!   $\mathcal{N}(\mu, \sigma^2)$ via the existing [`SimdNormal`](crate::normal::SimdNormal) sampler
//!   while acceptance stays above 5 %. Tight intervals (mass $< 0.05$,
//!   where rejection would spin) split by where they sit: one lying wholly
//!   on one side of the mean takes Robert's accept-reject on the
//!   standardised interval, which never forms a CDF and so cannot lose the
//!   interval's mass to rounding; one straddling the mean takes the
//!   closed-form inverse-CDF transform, exact where the normal CDF still
//!   resolves, at one [`crate::special::ndtri`] call per draw.
//! - **Truncated Exponential:** closed-form inverse-CDF sampling on the
//!   survival function — no rejection needed.
//! - **Truncated Beta / Gamma:** plain rejection from the corresponding
//!   [`SimdBeta`](crate::beta::SimdBeta) / [`SimdGamma`](crate::gamma::SimdGamma) sampler. For very tight intervals where
//!   acceptance falls below 1 % the rejection loop bails after 1000 tries
//!   and returns the clamped midpoint — the caller should widen the
//!   bounds in that regime (the boundary itself is hit with measure zero).
//!
//! ## Density
//!
//! Standard normalisation: $f_{\[a,b\]}(x) = f(x) / (F(b) - F(a))$ for
//! $x \in [a, b]$ and $0$ elsewhere. The CDF normalising constant is
//! cached at construction time.
//!
//! References:
//! - Devroye, L. (1986), *Non-Uniform Random Variate Generation*,
//!   Springer, §II.2 (inversion, the exponential) and §II.3 (rejection,
//!   the wide-interval path of the other three).
//! - Robert, C.P. (1995), "Simulation of truncated normal variables",
//!   *Statistics and Computing* 5(2), 121-125, DOI: 10.1007/BF00143942
//!   (the two one-sided proposals the tail path picks between).

mod beta_gamma;
mod exp;
mod normal;

pub use beta_gamma::SimdTruncatedBeta;
pub use beta_gamma::SimdTruncatedGamma;
pub use exp::SimdTruncatedExp;
#[doc(hidden)]
pub use exp::TruncatedExpState;
pub use normal::SimdTruncatedNormal;
#[doc(hidden)]
pub use normal::TruncatedNormalState;
