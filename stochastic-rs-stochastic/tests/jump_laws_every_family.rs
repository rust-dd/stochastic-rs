use std::any::Any;

use rand::distr::Distribution;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::alpha_stable::SimdAlphaStable;
use stochastic_rs_distributions::cauchy::SimdCauchy;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::studentt::SimdStudentT;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::traits::ProcessExt;

const SEED: u64 = 17;
const PATHS: usize = 96;
const STEPS: usize = 32;

fn paths<D>(law: D, threads: usize) -> Vec<Vec<u64>>
where
  D: Distribution<f64> + Send + Sync + Any,
{
  ThreadPoolBuilder::new()
    .num_threads(threads)
    .build()
    .expect("rayon pool")
    .install(|| {
      Merton::new(
        0.03,
        0.2,
        30.0,
        0.0,
        law,
        STEPS,
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED),
      )
      .sample_par(PATHS)
      .iter()
      .map(|path| path.iter().map(|x| x.to_bits()).collect())
      .collect()
    })
}

fn is_a_jump_law<D>(name: &str, law: D)
where
  D: Distribution<f64> + Send + Sync + Any + Clone,
{
  let one = paths(law.clone(), 1);
  let eight = paths(law, 8);
  assert_eq!(
    one, eight,
    "{name}: sample_par differs between 1 and 8 threads"
  );
  assert!(
    one.iter().flatten().all(|b| f64::from_bits(*b).is_finite()),
    "{name}: a path is not finite"
  );
}

#[test]
fn every_stateless_family_is_a_jump_law() {
  is_a_jump_law("Normal", SimdNormal::<f64>::new(0.0, 0.1));
  is_a_jump_law("Exp", SimdExp::<f64>::new(5.0));
  is_a_jump_law("Gamma", SimdGamma::<f64>::new(2.0, 0.05));
  is_a_jump_law("StudentT", SimdStudentT::<f64>::new(5.0));
  is_a_jump_law(
    "AlphaStable",
    SimdAlphaStable::<f64>::new(1.7, 0.2, 0.05, 0.0),
  );
  is_a_jump_law("Cauchy", SimdCauchy::<f64>::new(0.0, 0.01));
}
