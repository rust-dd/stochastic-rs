//! The Riemann–Liouville Markov lift: the history as a bounded state of `N' ≈ log n` factors.
//!
//! $$
//! \begin{aligned}
//! X_{n+1} &= X_0 + \frac{\delta t^{H+1/2} f(X_n)}{\Gamma(H+3/2)} +
//! \frac{\sum_{l=1}^{N'} w_l\, e^{-x_l \delta t}\,(H_l^{(n)} + J_l^{(n)})}{\Gamma(H+1/2)} +
//! \frac{\delta t^{H-1/2} g(X_n)\,\delta W_n}{\Gamma(H+1/2)} \\
//! H_l^{(n+1)} &= \tfrac{f(X_n)}{x_l}\bigl(1 - e^{-x_l \delta t}\bigr) +
//! e^{-x_l \delta t}\, H_l^{(n)} \\
//! J_l^{(n+1)} &= e^{-x_l \delta t}\bigl(g(X_n)\,\delta W_n + J_l^{(n)}\bigr)
//! \end{aligned}
//! $$
//!
//! [`VolterraLift`](crate::volterra::VolterraLift) for `RlKernel<T>` and $(x)$-only coefficients.
//!
//! Reference: Bilokon & Wong (2026), J. Appl. Probab., p. 16

mod stepper;

#[cfg(test)]
mod tests;

pub use stepper::MarkovLift;
