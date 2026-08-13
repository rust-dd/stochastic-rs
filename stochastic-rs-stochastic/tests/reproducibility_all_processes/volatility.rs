//! `volatility/` slice of the exhaustive reproducibility guard — see
//! `../reproducibility_all_processes.rs` for the full rationale, the
//! derivation of the 127-type list, and shared methodology notes.
//! `RoughHeston::new` does not accept `rho` — matching this crate's own
//! `deterministic_parallelism_bates_rough_heston.rs`, the guard builds it
//! then sets the public field directly before sampling.

use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::bates_svj::BatesSvj;
use stochastic_rs_stochastic::volatility::bergomi::Bergomi;
use stochastic_rs_stochastic::volatility::bns::Bns;
use stochastic_rs_stochastic::volatility::double_heston::DoubleHeston;
use stochastic_rs_stochastic::volatility::fbates_svj::FBatesSvj;
use stochastic_rs_stochastic::volatility::fheston::RoughHeston;
use stochastic_rs_stochastic::volatility::heston::Heston;
use stochastic_rs_stochastic::volatility::heston_log::HestonLog;
use stochastic_rs_stochastic::volatility::heston2d::Heston2D;
use stochastic_rs_stochastic::volatility::hkde::Hkde;
use stochastic_rs_stochastic::volatility::multifactor_heston::MultifactorHeston;
use stochastic_rs_stochastic::volatility::multifactor_sabr::MultifactorSabr;
use stochastic_rs_stochastic::volatility::rbergomi::RoughBergomi;
use stochastic_rs_stochastic::volatility::sabr::Sabr;
use stochastic_rs_stochastic::volatility::svcgmy::Svcgmy;

use crate::common::J;
use crate::common::N;
use crate::common::guard;

guard!(bates_svj, "BatesSvj", |s| BatesSvj::new(
  Some(0.05),
  None,
  None,
  None,
  0.5,
  -0.1,
  0.2,
  0.04,
  1.5,
  0.3,
  -0.6,
  N,
  Some(100.0),
  Some(0.04),
  Some(1.0),
  Some(false),
  s
));

guard!(bergomi, "Bergomi", |s| Bergomi::new(
  0.4,
  Some(0.2),
  Some(100.0),
  0.01,
  -0.6,
  N,
  Some(1.0),
  s
));

guard!(bns, "Bns", |s| Bns::new(
  Some(100.0),
  0.04,
  0.1,
  0.0,
  1.0,
  1.0,
  N,
  Some(1.0),
  s
));

guard!(double_heston, "DoubleHeston", |s| DoubleHeston::new(
  Some(100.0),
  Some(0.02),
  Some(0.02),
  3.0,
  0.02,
  0.4,
  -0.6,
  0.5,
  0.02,
  0.2,
  -0.3,
  0.05,
  N,
  Some(1.0),
  Some(true),
  s
));

guard!(f_bates_svj, "FBatesSvj", |s| FBatesSvj::new(
  0.1,
  0.05,
  100.0,
  0.04,
  0.04,
  2.0,
  0.3,
  -0.7,
  0.5,
  -0.01,
  0.1,
  N,
  Some(1.0),
  s
));

guard!(rough_heston, "RoughHeston", |s| {
  let mut m = RoughHeston::new(0.1, Some(0.04), 0.04, 1.5, 0.3, None, None, Some(1.0), N, s);
  m.rho = Some(-0.6);
  m
});

guard!(heston, "Heston", |s| Heston::new(
  Some(100.0),
  Some(0.04),
  2.0,
  0.04,
  0.3,
  -0.7,
  0.05,
  N,
  Some(1.0),
  HestonPow::Sqrt,
  Some(false),
  s
));

guard!(heston2d, "Heston2D", |s| Heston2D::new(
  [Some(0.0), Some(0.0)],
  [Some(0.4), Some(0.4)],
  [0.0, 0.0],
  [0.4, 0.4],
  [2.0, 2.0],
  [1.0, 1.0],
  [0.5, -0.5, 0.0, 0.0, -0.5, 0.5],
  N,
  Some(1.0),
  Some(false),
  s
));

guard!(heston_log, "HestonLog", |s| HestonLog::new(
  Some(0.05),
  None,
  None,
  None,
  1.5,
  0.04,
  0.3,
  -0.7,
  N,
  Some(100.0),
  Some(0.04),
  Some(1.0),
  Some(false),
  s
));

guard!(hkde, "Hkde", |s| Hkde::new(
  0.05,
  1.5,
  0.04,
  0.3,
  -0.7,
  0.04,
  0.5,
  0.4,
  5.0,
  5.0,
  N,
  Some(100.0),
  Some(1.0),
  Some(false),
  s
));

guard!(multifactor_heston, "MultifactorHeston", |s| {
  MultifactorHeston::new(
    Some(100.0),
    [0.04, 0.04],
    [1.5, 1.5],
    [0.04, 0.04],
    [0.3, 0.3],
    [-0.5, -0.5],
    0.0,
    N,
    Some(1.0),
    s,
  )
});

guard!(multifactor_sabr, "MultifactorSabr", |s| {
  MultifactorSabr::new(
    Some(100.0),
    Some(0.3),
    vec![0.5],
    vec![0.5, 0.5],
    vec![0.0, 0.0],
    vec![0.3, 0.3],
    N,
    Some(1.0),
    s,
  )
});

guard!(rough_bergomi, "RoughBergomi", |s| RoughBergomi::new(
  0.1,
  0.4,
  Some(0.2),
  Some(100.0),
  0.01,
  -0.6,
  N,
  Some(1.0),
  s
));

guard!(sabr, "Sabr", |s| Sabr::new(
  0.4,
  0.7,
  -0.3,
  N,
  Some(1.0),
  Some(0.3),
  Some(1.0),
  s
));

guard!(svcgmy, "Svcgmy", |s| Svcgmy::new(
  1.0,
  1.0,
  0.5,
  2.0,
  0.04,
  0.2,
  0.0,
  N,
  J,
  Some(0.0),
  Some(0.04),
  Some(1.0),
  s
));
