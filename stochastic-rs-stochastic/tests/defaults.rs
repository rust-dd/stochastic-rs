//! TDD tests for A1-c Task 1: `Default` + `Clone` on the flagship process
//! types. See `stochastic-rs-stochastic/src/traits/process.rs`'s `## Clone
//! semantics` section for the pinned decision
//! `clone_preserves_deterministic_path` below exercises, and each type's own
//! `Default` impl doc for where its parameter values come from.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
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
use stochastic_rs_stochastic::volatility::HestonPow;
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
  // `Kou` has no `Default` — see `Kou`'s own struct doc: a Gaussian jump
  // distribution would silently ship Merton-with-Gaussian-jumps under the
  // Kou name, since only `D` distinguishes the two samplers in this crate.

  let [s, v] = Heston::<f64>::default().sample();
  assert!(ok(&s) && ok(&v));
  let [f_, a] = Sabr::<f64>::default().sample();
  assert!(ok(&f_) && ok(&a));
  let [s, v2] = Bergomi::<f64>::default().sample();
  assert!(ok(&s) && ok(&v2));
  let [s, v2] = RoughBergomi::<f64>::default().sample();
  assert!(ok(&s) && ok(&v2));
}

/// `CirPlusPlus::default()` must clear the Feller guard in real `f64`
/// arithmetic — not merely avoid the warning via `use_sym`, which would
/// leave the type sub-Feller in fact while only suppressing the print. See
/// `CirPlusPlus::default`'s own doc for why its previous parameterization
/// (κ=0.5) failed this by one ulp (`2·0.5·0.04 == 0.04` but
/// `0.2 * 0.2 == 0.040000000000000001`).
#[test]
fn cir_pp_default_clears_feller_guard() {
  let d = CirPlusPlus::<f64>::default();
  assert!(2.0 * d.kappa * d.theta >= d.sigma * d.sigma);
}

fn bits(path: &Array1<f64>) -> Vec<u64> {
  path.iter().map(|x| x.to_bits()).collect()
}

fn theta04(_t: f64) -> f64 {
  0.04
}

fn zero_phi(_t: f64) -> f64 {
  0.0
}

fn theta05(_t: f64) -> f64 {
  0.05
}

/// Same parameter values as each type's `Default` (seed swapped for an
/// explicit `Deterministic` one) — `x.clone().sample() == x.sample()`
/// bit-for-bit must hold uniformly per the pinned decision (`ProcessExt`'s
/// `## Clone semantics`): whole-struct `Clone` snapshots the seed rather
/// than forking it, so the clone replays the identical path if sampled
/// before either side draws anything else.
///
/// `Merton`/`Kou` are included below (they were excluded before the
/// zero-exception-reproducibility wave's Task 1: their `cpoisson` field
/// type was `CompoundPoisson<T, D>`, fixing *that* field's own seed
/// strategy to `Unseeded` regardless of the outer process's `S`, so
/// `x.sample() != x.sample()` held for the very same instance before
/// `Clone` even entered the picture). `new()` now builds `cpoisson`
/// internally from the same `seed: S` passed in, so `#[derive(Clone)]`'s
/// plain field-wise clone of `seed` — the mechanism this test otherwise
/// exercises uniformly — snapshots the jump component's reproducibility
/// exactly as it always did the diffusion's. A nonzero-intensity jump
/// distribution is used for both so the jump half is actually exercised,
/// not short-circuited to an all-zero, RNG-free array.
#[test]
fn clone_preserves_deterministic_path() {
  let a = Gbm::<f64, _>::new(0.05, 0.2, N, Some(100.0), Some(1.0), Deterministic::new(42));
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = Ou::<f64, _>::new(
    2.0,
    0.0,
    0.2,
    N,
    Some(0.0),
    Some(1.0),
    Deterministic::new(42),
  );
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = Cir::<f64, _>::new(
    2.5,
    0.04,
    0.2,
    N,
    Some(0.04),
    Some(1.0),
    Some(false),
    Deterministic::new(42),
  );
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = Cev::<f64, _>::new(
    0.04,
    0.2,
    0.8,
    N,
    Some(1.0),
    Some(1.0),
    Deterministic::new(42),
  );
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = Vg::<f64, _>::new(
    0.0,
    0.2,
    0.15,
    N,
    Some(0.0),
    Some(1.0),
    Deterministic::new(42),
  );
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = Vasicek::<f64, _>::new(
    3.0,
    0.03,
    0.02,
    N,
    Some(0.03),
    Some(1.0),
    Deterministic::new(42),
  );
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = HullWhite::<f64, _>::new(
    theta04 as fn(f64) -> f64,
    0.4,
    0.02,
    N,
    Some(0.02),
    Some(1.0),
    Deterministic::new(42),
  );
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = CirPlusPlus::<f64, _>::new(
    2.5,
    0.04,
    0.2,
    zero_phi as fn(f64) -> f64,
    N,
    Some(0.04),
    Some(1.0),
    None,
    Deterministic::new(42),
  );
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = BlackKarasinski::<f64, _>::new(
    theta05 as fn(f64) -> f64,
    0.8,
    0.1,
    N,
    Some(0.03),
    Some(1.0),
    Deterministic::new(42),
  );
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = Fgn::<f64, _>::new(0.7, N, Some(1.0), Deterministic::new(42));
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = Gn::<f64, _>::new(N, Some(1.0), Deterministic::new(42));
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = Bm::<f64, _>::new(N, Some(1.0), Deterministic::new(42));
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = Fbm::<f64, _>::new(0.7, N, Some(1.0), Deterministic::new(42));
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = Poisson::<f64, _>::new(2.0, Some(N), Some(1.0), Deterministic::new(42));
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = BrownianBridge::<f64, _>::new(1.0, N, None, None, Some(1.0), Deterministic::new(42));
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = SquaredBessel::<f64, _>::new(3.0, N, Some(1.0), Some(1.0), None, Deterministic::new(42));
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = Bessel::<f64, _>::new(3.0, N, Some(1.0), Some(1.0), None, Deterministic::new(42));
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = DisplacedDiffusion::<f64, _>::new(
    0.05,
    0.2,
    30.0,
    N,
    Some(100.0),
    Some(1.0),
    Deterministic::new(42),
  );
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = Heston::<f64, _>::new(
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
    Deterministic::new(42),
  );
  let b = a.clone();
  let [s1, v1] = a.sample();
  let [s2, v2] = b.sample();
  assert_eq!(bits(&s1), bits(&s2));
  assert_eq!(bits(&v1), bits(&v2));

  let a = Sabr::<f64, _>::new(
    0.4,
    0.7,
    -0.3,
    N,
    Some(1.0),
    Some(0.3),
    Some(1.0),
    Deterministic::new(42),
  );
  let b = a.clone();
  let [f1, a1] = a.sample();
  let [f2, a2] = b.sample();
  assert_eq!(bits(&f1), bits(&f2));
  assert_eq!(bits(&a1), bits(&a2));

  let a = Bergomi::<f64, _>::new(
    0.4,
    Some(0.2),
    Some(100.0),
    0.01,
    -0.6,
    N,
    Some(1.0),
    Deterministic::new(42),
  );
  let b = a.clone();
  let [s1, v1] = a.sample();
  let [s2, v2] = b.sample();
  assert_eq!(bits(&s1), bits(&s2));
  assert_eq!(bits(&v1), bits(&v2));

  let a = RoughBergomi::<f64, _>::new(
    0.1,
    0.4,
    Some(0.2),
    Some(100.0),
    0.01,
    -0.6,
    N,
    Some(1.0),
    Deterministic::new(42),
  );
  let b = a.clone();
  let [s1, v1] = a.sample();
  let [s2, v2] = b.sample();
  assert_eq!(bits(&s1), bits(&s2));
  assert_eq!(bits(&v1), bits(&v2));

  // `Merton`/`Kou`: see this function's module doc for why these two were
  // excluded before the zero-exception-reproducibility wave's Task 1, and
  // why a nonzero-intensity jump distribution is used here (proves the
  // jump component specifically, not just the diffusion).
  let a = Merton::new(
    0.03,
    0.2,
    1.0,
    0.0,
    ScalarNormal::new(0.0, 0.1),
    N,
    Some(0.0),
    Some(1.0),
    Deterministic::new(42),
  );
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));

  let a = Kou::new(
    0.03,
    0.2,
    1.0,
    0.0,
    ScalarNormal::new(0.0, 0.12),
    N,
    Some(0.0),
    Some(1.0),
    Deterministic::new(42),
  );
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));
}
