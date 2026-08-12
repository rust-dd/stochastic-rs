mod categories;
mod plot_grid;
mod sde_methods;
mod smoke;

pub(crate) fn f_const_001(_: f64) -> f64 {
  0.01
}

pub(crate) fn f_const_002(_: f64) -> f64 {
  0.02
}

pub(crate) fn f_linear_small(t: f64) -> f64 {
  0.01 + 0.005 * t
}

pub(crate) fn f_phi_small(t: f64) -> f64 {
  0.002 * t
}

pub(crate) fn f_hjm_p(t: f64, u: f64) -> f64 {
  0.01 + 0.01 * (u - t).max(0.0)
}

pub(crate) fn f_hjm_q(_: f64, _: f64) -> f64 {
  0.5
}

pub(crate) fn f_hjm_v(_: f64, _: f64) -> f64 {
  0.02
}

pub(crate) fn f_hjm_alpha(_: f64, _: f64) -> f64 {
  0.01
}

pub(crate) fn f_hjm_sigma(_: f64, _: f64) -> f64 {
  0.015
}

pub(crate) fn f_adg_k(t: f64) -> f64 {
  0.02 + 0.002 * t
}

pub(crate) fn f_adg_theta(_: f64) -> f64 {
  0.6
}

pub(crate) fn f_adg_phi(_: f64) -> f64 {
  0.01
}

pub(crate) fn f_adg_b(_: f64) -> f64 {
  0.2
}

pub(crate) fn f_adg_c(_: f64) -> f64 {
  0.05
}
