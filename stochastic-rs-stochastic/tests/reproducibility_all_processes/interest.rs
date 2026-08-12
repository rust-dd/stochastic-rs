//! `interest/` slice of the exhaustive reproducibility guard — see
//! `../reproducibility_all_processes.rs` for the full rationale, the
//! derivation of the 124-type list, and shared methodology notes. Six of
//! these types (`Adg`, `BlackKarasinski`, `Cir2F`'s own `phi`, `CirPlusPlus`,
//! `Hjm`, `HullWhite`, `HullWhite2F`) take an `impl Into<Fn1D<T>>`/
//! `Fn2D<T>` curve parameter; a bare closure does not coerce to that bound
//! on its own, so every one is passed as a named function cast explicitly
//! to a function pointer (`fn1d_a as fn(f64) -> f64`), the same pattern
//! this crate's own `with_setters_*` tests use for the same reason.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
// `Cir2F` needs two full `Cir<T, S>` sub-processes — the crate's most
// constructor-heavy type — pulled in from the diffusion family.
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::interest::adg::Adg;
use stochastic_rs_stochastic::interest::bgm::Bgm;
use stochastic_rs_stochastic::interest::black_karasinski::BlackKarasinski;
use stochastic_rs_stochastic::interest::cir_2f::Cir2F;
use stochastic_rs_stochastic::interest::cir_pp::CirPlusPlus;
use stochastic_rs_stochastic::interest::duffie_kan::DuffieKan;
use stochastic_rs_stochastic::interest::duffie_kan_jump_exp::DuffieKanJumpExp;
use stochastic_rs_stochastic::interest::fractional_vasicek::FVasicek;
use stochastic_rs_stochastic::interest::hjm::Hjm;
use stochastic_rs_stochastic::interest::ho_lee::HoLee;
use stochastic_rs_stochastic::interest::hull_white::HullWhite;
use stochastic_rs_stochastic::interest::hull_white_2f::HullWhite2F;
use stochastic_rs_stochastic::interest::lmm::Lmm;
use stochastic_rs_stochastic::interest::vasicek::Vasicek;
use stochastic_rs_stochastic::interest::wu_zhang::WuZhangD;

use crate::common::N;
use crate::common::fn1d_a;
use crate::common::fn2d_a;
use crate::common::guard;

guard!(adg, "Adg", |s| Adg::new(
  fn1d_a as fn(f64) -> f64,
  fn1d_a as fn(f64) -> f64,
  Array1::from(vec![0.01, 0.01, 0.01]),
  fn1d_a as fn(f64) -> f64,
  fn1d_a as fn(f64) -> f64,
  fn1d_a as fn(f64) -> f64,
  N,
  3,
  Array1::from(vec![0.03, 0.03, 0.03]),
  Some(1.0),
  s
));

guard!(bgm, "Bgm", |s| Bgm::new(
  Array1::from(vec![0.2, 0.2, 0.2]),
  Array1::from(vec![0.03, 0.03, 0.03]),
  3,
  Some(1.0),
  N,
  s
));

guard!(black_karasinski, "BlackKarasinski", |s| {
  BlackKarasinski::new(
    fn1d_a as fn(f64) -> f64,
    0.1,
    0.02,
    N,
    Some(0.03),
    Some(1.0),
    s,
  )
});

guard!(cir_2f, "Cir2F", |s: Deterministic| Cir2F::new(
  Cir::new(
    1.0,
    0.03,
    0.1,
    N,
    Some(0.03),
    Some(1.0),
    Some(false),
    s.clone()
  ),
  Cir::new(
    1.2,
    0.02,
    0.1,
    N,
    Some(0.02),
    Some(1.0),
    Some(false),
    s.clone()
  ),
  fn1d_a as fn(f64) -> f64,
  s
));

guard!(cir_plus_plus, "CirPlusPlus", |s| CirPlusPlus::new(
  1.0,
  0.04,
  0.2,
  fn1d_a as fn(f64) -> f64,
  N,
  Some(0.03),
  Some(1.0),
  Some(false),
  s
));

guard!(duffie_kan, "DuffieKan", |s| DuffieKan::new(
  0.5,
  0.04,
  0.5,
  -0.3,
  0.01,
  0.0,
  0.0,
  0.01,
  0.0,
  0.5,
  0.0,
  0.005,
  N,
  Some(0.05),
  Some(0.05),
  Some(1.0),
  s
));

guard!(duffie_kan_jump_exp, "DuffieKanJumpExp", |s| {
  DuffieKanJumpExp::new(
    0.5,
    0.04,
    0.5,
    -0.3,
    0.01,
    0.0,
    0.0,
    0.01,
    0.0,
    0.5,
    0.0,
    0.005,
    0.5,
    0.01,
    N,
    Some(0.05),
    Some(0.05),
    Some(1.0),
    s,
  )
});

guard!(f_vasicek, "FVasicek", |s| FVasicek::new(
  0.7,
  1.0,
  0.03,
  0.02,
  N,
  Some(0.03),
  Some(1.0),
  s
));

guard!(hjm, "Hjm", |s| Hjm::new(
  fn1d_a as fn(f64) -> f64,
  fn1d_a as fn(f64) -> f64,
  fn2d_a as fn(f64, f64) -> f64,
  fn2d_a as fn(f64, f64) -> f64,
  fn2d_a as fn(f64, f64) -> f64,
  fn2d_a as fn(f64, f64) -> f64,
  fn2d_a as fn(f64, f64) -> f64,
  N,
  Some(0.03),
  Some(1.0),
  Some(0.03),
  Some(1.0),
  s
));

guard!(ho_lee, "HoLee", |s| HoLee::new(
  None,
  Some(0.02),
  0.01,
  N,
  Some(1.0),
  s
));

guard!(hull_white, "HullWhite", |s| HullWhite::new(
  fn1d_a as fn(f64) -> f64,
  0.1,
  0.01,
  N,
  Some(0.03),
  Some(1.0),
  s
));

guard!(hull_white_2f, "HullWhite2F", |s| HullWhite2F::new(
  fn1d_a as fn(f64) -> f64,
  0.1,
  0.01,
  0.01,
  0.2,
  0.05,
  Some(0.03),
  Some(1.0),
  N,
  s
));

guard!(lmm, "Lmm", |s| Lmm::new(
  Array1::from(vec![0.0, 0.5, 1.0, 1.5, 2.0]),
  Array1::from(vec![0.03, 0.035, 0.04, 0.045]),
  Array1::from(vec![0.20, 0.20, 0.20, 0.20]),
  N,
  Some(2.0),
  s
));

guard!(vasicek, "Vasicek", |s| Vasicek::new(
  3.0,
  0.03,
  0.02,
  N,
  Some(0.03),
  Some(1.0),
  s
));

guard!(wu_zhang, "WuZhangD", |s| WuZhangD::new(
  Array1::from(vec![0.1, 0.1]),
  Array1::from(vec![0.1, 0.1]),
  Array1::from(vec![0.2, 0.2]),
  Array1::from(vec![0.1, 0.1]),
  Array1::from(vec![100.0, 100.0]),
  Array1::from(vec![0.04, 0.04]),
  2,
  Some(1.0),
  N,
  s
));
