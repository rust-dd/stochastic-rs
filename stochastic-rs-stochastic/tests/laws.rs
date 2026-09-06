//! Every process against the closed form its own source states.
//!
//! Distinct from `device_law`, which holds a device to what this crate's CPU
//! sampler produces: a case here is a statement of the *model* — a published
//! moment, autocorrelation or boundary — so a sampler that is wrong on both
//! host and device still fails. The two literature bugs found in September
//! 2026 (the fractional Brownian field's correction term, the tempered-stable
//! series divisor) were both invisible to a host-device comparison and both
//! fell out of a closed form.
//!
//! Tolerances are not chosen: every case estimates a statistic per path and
//! compares its mean to the closed form through the standard error across
//! paths, which is what `common` exists for.

mod laws {
  pub(crate) mod autoregressive;
  pub(crate) mod common;
}
