//! # Autoregressive
//!
//! $$
//! dX_t=a(t,X_t)dt+b(t,X_t)dW_t
//! $$
//!
pub mod agrach;
pub mod ar;
pub mod arch;
pub mod arima;
pub mod egarch;
pub mod garch;
pub mod ma;
pub mod sarima;
pub mod tgarch;