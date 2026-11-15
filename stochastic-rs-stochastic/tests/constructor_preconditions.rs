use ndarray::array;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::correlation::stochastic_correlated_bm;
use stochastic_rs_stochastic::noise::cfgns::Cfgns;

#[test]
#[should_panic(expected = "rho must satisfy `")]
fn a_correlated_fgn_pair_names_the_predicate_it_broke() {
  let _ = Cfgns::<f64>::new(0.7, 1.5, 16, None, Unseeded);
}

#[test]
#[should_panic(
  expected = "rho_path[1] must satisfy `(-1.0..=1.0).contains(&rho_path[i])`, got rho_path[1] = 1.2"
)]
fn correlated_brownian_motions_refuse_a_correlation_outside_the_unit_interval() {
  let _ = stochastic_correlated_bm(&array![0.2, 1.2, 0.1], None);
}

#[test]
#[should_panic(
  expected = "rho_path[0] must satisfy `(-1.0..=1.0).contains(&rho_path[i])`, got rho_path[0] = NaN"
)]
fn correlated_brownian_motions_refuse_a_nan_correlation() {
  let _ = stochastic_correlated_bm(&array![f64::NAN, 0.1], None);
}
