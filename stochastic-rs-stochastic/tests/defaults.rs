//! TDD tests for A1-c Task 1: `Default` + `Clone` on the flagship process
//! types. See `stochastic-rs-stochastic/src/traits/process.rs`'s `## Clone
//! semantics` section for the pinned decision
//! `clone_preserves_deterministic_path` below exercises, and each type's own
//! `Default` impl doc for where its parameter values come from.

use ndarray::Array1;
use stochastic_rs_distributions::scalar::ScalarNormal;
use stochastic_rs_stochastic::diffusion::bessel::Bessel;
use stochastic_rs_stochastic::diffusion::bessel::SquaredBessel;
use stochastic_rs_stochastic::diffusion::cev::Cev;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::displaced_diffusion::DisplacedDiffusion;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::interest::black_karasinski::BlackKarasinski;
use stochastic_rs_stochastic::interest::cir_pp::CirPlusPlus;
use stochastic_rs_stochastic::interest::hull_white::HullWhite;
use stochastic_rs_stochastic::interest::vasicek::Vasicek;
use stochastic_rs_stochastic::jump::kou::Kou;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::jump::vg::Vg;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::noise::gn::Gn;
use stochastic_rs_stochastic::process::bm::Bm;
use stochastic_rs_stochastic::process::brownian_bridge::BrownianBridge;
use stochastic_rs_stochastic::process::fbm::Fbm;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::bergomi::Bergomi;
use stochastic_rs_stochastic::volatility::heston::Heston;
use stochastic_rs_stochastic::volatility::rbergomi::RoughBergomi;
use stochastic_rs_stochastic::volatility::sabr::Sabr;

const N: usize = 252;

fn ok(path: &Array1<f64>) -> bool {
  path.len() == N && path.iter().all(|x| x.is_finite())
}

/// Every Default-constructible process must sample finite output out of the
/// box, at its documented default length (`n = 252`).
#[test]
fn defaults_sample_finite() {
  assert!(ok(&Gbm::<f64>::default().sample()));
  assert!(ok(&Ou::<f64>::default().sample()));
  assert!(ok(&Cir::<f64>::default().sample()));
  assert!(ok(&Cev::<f64>::default().sample()));
  assert!(ok(&Vg::<f64>::default().sample()));
  assert!(ok(&Vasicek::<f64>::default().sample()));
  assert!(ok(&HullWhite::<f64>::default().sample()));
  assert!(ok(&CirPlusPlus::<f64>::default().sample()));
  assert!(ok(&BlackKarasinski::<f64>::default().sample()));
  assert!(ok(&Fgn::<f64>::default().sample()));
  assert!(ok(&Gn::<f64>::default().sample()));
  assert!(ok(&Bm::<f64>::default().sample()));
  assert!(ok(&Fbm::<f64>::default().sample()));
  assert!(ok(&Poisson::<f64>::default().sample()));
  assert!(ok(&BrownianBridge::<f64>::default().sample()));
  assert!(ok(&SquaredBessel::<f64>::default().sample()));
  assert!(ok(&Bessel::<f64>::default().sample()));
  assert!(ok(&DisplacedDiffusion::<f64>::default().sample()));
  assert!(ok(&Merton::<f64, ScalarNormal<f64>>::default().sample()));
  assert!(ok(&Kou::<f64, ScalarNormal<f64>>::default().sample()));

  let [s, v] = Heston::<f64>::default().sample();
  assert!(ok(&s) && ok(&v));
  let [f_, a] = Sabr::<f64>::default().sample();
  assert!(ok(&f_) && ok(&a));
  let [s, v2] = Bergomi::<f64>::default().sample();
  assert!(ok(&s) && ok(&v2));
  let [s, v2] = RoughBergomi::<f64>::default().sample();
  assert!(ok(&s) && ok(&v2));
}
