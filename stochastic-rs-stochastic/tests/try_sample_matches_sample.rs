//! `ProcessExt::try_sample` / `try_sample_par` on the host devices are the
//! plain `sample` / `sample_par`: always `Ok`, bit-identical paths, the same
//! seed consumption. One process per sampling shape: a host sampler (`Bm`),
//! the Euler engine (`Ou`), the fGN generator (`Fgn`), a wrapper that batches
//! through it (`Fbm`), one that draws per path (`Fou`) and one that draws a
//! pair (`Cfgns`).

use std::fmt::Debug;

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::noise::cfgns::Cfgns;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::process::bm::Bm;
use stochastic_rs_stochastic::process::fbm::Fbm;
use stochastic_rs_stochastic::traits::ProcessExt;

fn fallible_is_plain<P: ProcessExt<f64>>(make: impl Fn() -> P)
where
  P::Output: PartialEq + Debug,
{
  assert_eq!(make().try_sample().expect("host"), make().sample());
  for m in [0, 1, 2, 7, 130] {
    assert_eq!(
      make().try_sample_par(m).expect("host"),
      make().sample_par(m),
      "m = {m}"
    );
  }
  let fallible = make();
  let plain = make();
  let _ = fallible.try_sample().expect("host");
  let _ = plain.sample();
  assert_eq!(
    fallible.try_sample_par(3).expect("host"),
    plain.sample_par(3)
  );
}

#[test]
fn host_sampler_bm() {
  fallible_is_plain(|| Bm::<f64, _>::new(64, Some(1.0), Deterministic::new(7)));
}

#[test]
fn euler_engine_ou() {
  fallible_is_plain(|| {
    Ou::<f64, _>::new(
      0.5,
      0.02,
      0.1,
      64,
      Some(0.03),
      Some(1.0),
      Deterministic::new(7),
    )
  });
}

#[test]
fn fgn_generator() {
  fallible_is_plain(|| Fgn::<f64, _>::new(0.7, 64, Some(1.0), Deterministic::new(7)));
}

#[test]
fn fgn_batch_wrapper_fbm() {
  fallible_is_plain(|| Fbm::<f64, _>::new(0.7, 64, Some(1.0), Deterministic::new(7)));
}

#[test]
fn fgn_per_path_wrapper_fou() {
  fallible_is_plain(|| {
    Fou::<f64, _>::new(
      0.7,
      0.5,
      0.0,
      0.2,
      64,
      None,
      Some(1.0),
      Deterministic::new(7),
    )
  });
}

#[test]
fn fgn_pair_wrapper_cfgns() {
  fallible_is_plain(|| Cfgns::<f64, _>::new(0.7, 0.3, 64, Some(1.0), Deterministic::new(7)));
}
