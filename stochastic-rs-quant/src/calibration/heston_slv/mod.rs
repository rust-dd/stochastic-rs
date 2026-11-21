//! Heston stochastic-local volatility calibration: a Heston fit, Dupire's local vol, then `L`:
//! $$
//! \begin{aligned}
//! \frac{dS_t}{S_t} &= (r-q)\,dt + L(t, S_t)\,\sqrt{V_t}\,dW_t^S,\\
//! dV_t &= \kappa(\theta - V_t)\,dt + \eta\sigma\sqrt{V_t}\,dW_t^V,
//! \qquad d\langle W^S, W^V\rangle_t = \rho\,dt
//! \end{aligned}
//! $$
//!
//! The mixing `η ∈ [0, 1]` is an input: it moves forward-start and barrier prices, never vanillas.
//! Reference: Guyon & Henry-Labordère (2012), Risk 25(1), 88–93
//! Reference: Lipton (2002), Risk 15(2), 61–65
//! Reference: van der Stoep et al. (2014), 17(7), DOI 10.1142/S0219024914500459

mod calibrator;

pub use calibrator::HestonSlvCalibrator;
pub use calibrator::HestonSlvFit;
pub use calibrator::LeverageMethod;

#[cfg(test)]
mod tests;
