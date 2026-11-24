//! Every process against the closed form its own source states, so a sampler wrong on both host
//! and device fails. Each case compares a per-path statistic's mean through its standard error.

mod laws {
  pub(crate) mod autoregressive;
  pub(crate) mod bounded;
  pub(crate) mod common;
  pub(crate) mod diffusion;
  pub(crate) mod inventory;
  pub(crate) mod levy;
  pub(crate) mod noise;
  pub(crate) mod short_rate;
  pub(crate) mod tempered_stable;
}
