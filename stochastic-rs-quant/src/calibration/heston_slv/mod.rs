//! # Heston stochastic-local volatility
//!
//! $$
//! \begin{aligned}
//! \frac{dS_t}{S_t} &= (r-q)\,dt + L(t, S_t)\,\sqrt{V_t}\,dW_t^S,\\
//! dV_t &= \kappa(\theta - V_t)\,dt + \eta\sigma\sqrt{V_t}\,dW_t^V,
//! \qquad d\langle W^S, W^V\rangle_t = \rho\,dt
//! \end{aligned}
//! $$
//!
//! From a vanilla call surface to the model in three steps, each of which
//! exists on its own in the crate and is wired here into one
//! [`Calibrator`](crate::traits::Calibrator):
//!
//! 1. **The Heston parameters** $(\kappa, \theta, \sigma, \rho, v_0)$ —
//!    fitted to the same quotes by
//!    [`HestonCalibrator`](crate::calibration::heston::HestonCalibrator), or
//!    pinned by the caller.
//! 2. **The Dupire local volatility** $\sigma_{\text{LV}}(t, K)$ — read off
//!    the call surface by [`Dupire`](crate::pricing::dupire::Dupire), or
//!    supplied on the same grid (an SSVI-derived surface, say).
//! 3. **The leverage function** $L(t, S)$ — the Guyon–Henry-Labordère
//!    particle method of
//!    [`calibrate_leverage`](crate::pricing::slv::calibrate_leverage), or the
//!    finite-volume solution of the forward Kolmogorov equation of
//!    [`calibrate_leverage_fokker_planck`](crate::pricing::slv::calibrate_leverage_fokker_planck)
//!    ([`LeverageMethod`]); either makes the model reproduce the
//!    local-volatility surface, and with it the vanillas, at any mixing
//!    fraction $\eta \in [0, 1]$.
//!
//! $\eta$ is an input, not an output: the vanilla surface is reproduced for
//! every value of it, and what it moves is the price of forward-starting and
//! barrier products, against which a desk chooses it. $\eta = 0$ is the pure
//! local-volatility model, $\eta = 1$ the Heston dynamics under a leverage
//! correction.
//!
//! The fit quality reported is the in-sample repricing of the input calls
//! from the calibration's own density at each maturity — the particle cloud's
//! payoff averages, or the trapezoid rule over the finite-volume marginal —
//! which is how the methods are judged in the literature.
//!
//! References: Guyon, J. & Henry-Labordère, P. (2012), *Being particular
//! about calibration*, Risk 25(1); Lipton, A. (2002), *The vol smile
//! problem*, Risk 15(2), 61–65; van der Stoep, A. W., Grzelak, L. A. &
//! Oosterlee, C. W. (2014), *The Heston stochastic-local volatility model:
//! efficient Monte Carlo simulation*, Int. J. Theor. Appl. Finance 17(7),
//! <https://doi.org/10.1142/S0219024914500459>.

mod calibrator;
mod result;

pub use calibrator::HestonSlvCalibrator;
pub use calibrator::LeverageMethod;
pub use result::HestonSlvCalibrationResult;
pub use result::HestonSlvFit;

#[cfg(test)]
mod tests;
