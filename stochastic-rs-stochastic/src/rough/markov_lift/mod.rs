//! # Markov-lift Volterra SDE stepper
//!
//! $$
//! \begin{aligned} X_{n+1} &= X_0 + \frac{\delta t^{H+1/2} f(X_n)}{\Gamma(H+3/2)} + \frac{\sum_{l=1}^{N'} w_l\, e^{-x_l \delta t}\,(H_l^{(n)} + J_l^{(n)})}{\Gamma(H+1/2)} + \frac{\delta t^{H-1/2} g(X_n)\,\delta W_n}{\Gamma(H+1/2)} \\ H_l^{(n+1)} &= \tfrac{f(X_n)}{x_l}\bigl(1 - e^{-x_l \delta t}\bigr) + e^{-x_l \delta t}\, H_l^{(n)} \\ J_l^{(n+1)} &= e^{-x_l \delta t}\bigl(g(X_n)\,\delta W_n + J_l^{(n)}\bigr) \end{aligned}
//! $$
//!
//! A Volterra-SDE simulator collapsing the full history into a bounded
//! state of $N' \approx \log n$ exponential factors. Provides two entry
//! points:
//!
//! - [`simulate`](MarkovLift::simulate) — single path, SIMD across the
//!   $N'$ quadrature factors via `wide::f64x4` (f64) / `wide::f32x8` (f32).
//! - [`simulate_batch`](MarkovLift::simulate_batch) — $m$ paths in one
//!   pass, SIMD across the *path* axis at each factor $l$ (BLAS-style
//!   batch parallelism, matches the layout of the Python reference
//!   `RoughHestonFast`).
//!
//! Load/store goes through unsafe pointer casts to avoid the element-wise
//! scatter-gather overhead of `f64x4::from([a, b, c, d])`. Reference:
//! Bilokon & Wong (2026), p. 16 of J. Appl. Probab. 2026.

mod simd;
mod stepper;

#[cfg(test)]
mod tests;

pub use simd::RoughSimd;
pub use stepper::BATCH_TILE;
pub use stepper::MarkovLift;
