//! Reference values are mpmath's at 40 digits: `gammainc` and `hyp1f1` for the primitives, the
//! rates from them and, independently, by quadrature of the remainder measure's tail.

use super::*;

fn close(got: f64, want: f64, rel: f64, what: &str) {
  assert!(
    (got - want).abs() <= rel * want.abs().max(1e-300),
    "{what}: got {got:e}, mpmath says {want:e}"
  );
}

#[test]
fn the_upper_gamma_function_reaches_non_positive_shapes() {
  for (c, x, want) in [
    (-1.5, 1e-6, 666664669.0289385),
    (-1.5, 0.05, 52.82510550261101),
    (-1.5, 0.7, 0.33333434409661183),
    (-1.5, 1.0, 0.12648781959325442),
    (-1.5, 3.0, 0.0018702598486750918),
    (-1.5, 40.0, 3.9565643509360977e-22),
    (-1.0, 1e-6, 999985.761704607),
    (-1.0, 0.05, 16.556690001504304),
    (-1.0, 0.7, 0.33563873361136165),
    (-1.0, 3.0, 0.0035473083617576103),
    (-0.5, 1e-6, 1996.4570922978558),
    (-0.5, 0.7, 0.3479027153786592),
    (-0.5, 40.0, 1.6199610039846915e-20),
    (0.0, 1e-6, 13.23829589306249),
    (0.0, 0.05, 2.467898488509974),
    (0.0, 1.0, 0.21938393439552026),
    (0.0, 3.0, 0.013048381094197037),
    (-0.75, 0.05, 9.657854194280635),
    (-0.75, 40.0, 6.403784286418288e-21),
  ] {
    let at_one = upper_gamma_fraction(c, 1.0);
    close(
      upper_gamma(c, x, at_one),
      want,
      1e-13,
      &format!("Γ({c}, {x})"),
    );
  }
}

#[test]
fn the_tempered_lower_integral_holds_across_the_pole_at_zero() {
  for (c, x, want) in [
    (-0.5, 1e-6, 0.001999999666666733),
    (-0.5, 0.05, 0.44352375315937437),
    (-0.5, 1.0, 1.7230554135925926),
    (-0.5, 3.0, 2.3969832994335505),
    (-0.5, 40.0, 3.228679935794194),
    (0.0, 1e-6, 9.999997500000556e-07),
    (0.0, 0.7, 0.5943095641963095),
    (0.0, 3.0, 1.6888763346638396),
    (0.0, 40.0, 4.2660951190154695),
    (0.5, 0.05, 0.007343074683892333),
    (0.5, 3.0, 1.717004273555702),
    (1.0, 1e-6, 4.999998333333749e-13),
    (1.0, 40.0, 39.0),
    (-0.9, 1e-6, 2.5118863173329453),
    (-0.9, 0.7, 9.37608563689685),
    (-0.9, 40.0, 10.530393955476596),
  ] {
    close(tempered_lower(c, x), want, 1e-13, &format!("L({c}, {x})"));
  }
}

#[test]
fn the_tempered_upper_integral_keeps_its_digits_below_one() {
  for (c, x, want) in [
    (-1.5, 1e-6, 1997.6377281986815),
    (-1.5, 0.05, 6.80337389738338),
    (-1.5, 1.0, 0.5401788470734122),
    (-1.5, 40.0, 0.0026352313834736496),
    (-1.0, 1e-6, 14.238295393062659),
    (-1.0, 0.7, 1.092932694960067),
    (-1.0, 3.0, 0.3297860249715757),
    (-0.5, 1e-6, 3.5429077021443653),
    (-0.5, 0.05, 3.101383948651658),
    (-0.5, 40.0, 0.31622776601683794),
    (-0.75, 0.7, 1.401924473707843),
    (-0.75, 3.0, 0.5800236480420334),
  ] {
    let at_one = upper_gamma_fraction(c, 1.0);
    close(
      tempered_upper(c, x, at_one),
      want,
      1e-13,
      &format!("M({c}, {x})"),
    );
  }
}

/// A remainder over a unit horizon from the two sides' scales `C₊`, `C₋`.
fn from_scales(alpha: f64, c: (f64, f64), lambda: (f64, f64), e_pow: f64) -> SeriesRemainder {
  let mass = c.0 + c.1;
  let e_scale = if e_pow == 1.0 {
    1.0
  } else {
    std::f64::consts::SQRT_2
  };
  SeriesRemainder::new(
    alpha,
    alpha / mass,
    1.0,
    c.0 / mass,
    lambda.0,
    lambda.1,
    e_scale,
    e_pow,
  )
}

#[test]
fn the_rates_are_the_moments_of_the_remainder_measure() {
  for (alpha, c, lambda, e_pow, eps, drift, variance) in [
    (
      1.5,
      (0.3, 0.3),
      (2.0, 6.0),
      1.0,
      0.01f64,
      0.3127023218759078,
      0.11729663416194686,
    ),
    (
      1.5,
      (0.5, 0.2),
      (2.0, 6.0),
      1.0,
      0.05,
      -2.6382257398678437,
      0.2897590769942013,
    ),
    (
      0.6,
      (0.5, 0.2),
      (2.0, 6.0),
      1.0,
      0.05,
      0.2128670513307104,
      0.005357219802568275,
    ),
    (
      1.7,
      (0.3, 0.3),
      (1.5, 4.0),
      0.5,
      0.2,
      0.28893261046665564,
      1.1763186786001834,
    ),
    (
      0.4,
      (0.3, 0.3),
      (1.5, 4.0),
      0.5,
      1.5,
      0.09756320560950064,
      0.08435833879194424,
    ),
    (
      1.2,
      (0.4, 0.1),
      (3.0, 1.0),
      1.0,
      2.0,
      -2.325469358874847,
      0.2729199586070246,
    ),
    (
      1.0,
      (0.3, 0.3),
      (2.0, 6.0),
      1.0,
      0.02,
      0.04224147621316099,
      0.010823757479898478,
    ),
    (
      1.9,
      (0.3, 0.3),
      (2.0, 6.0),
      1.0,
      50.0,
      9.988169569954739,
      5.048769836322385,
    ),
    (
      0.5,
      (0.5, 0.2),
      (0.001, 0.002),
      1.0,
      1e7,
      20.013974082506465,
      15855.001472750419,
    ),
    (
      1.5,
      (0.5, 0.2),
      (0.001, 0.002),
      1.0,
      1e7,
      -0.024343298842775456,
      35.95160073861474,
    ),
  ] {
    let remainder = from_scales(alpha, c, lambda, e_pow);
    let arrival = eps.powf(-alpha) / remainder.rate;
    let (got_drift, got_variance) = remainder.rates(arrival);
    let at = format!("α = {alpha}, C = {c:?}, λ = {lambda:?}, ε = {eps}");
    close(got_drift, drift, 1e-10, &format!("drift at {at}"));
    close(got_variance, variance, 1e-11, &format!("variance at {at}"));
  }
}

/// As `ε → 0` the tempering is flat and the remainder is the stable law's jumps below `ε`.
#[test]
fn a_small_cut_has_the_stable_small_jump_variance() {
  let (alpha, c) = (1.5, (0.3, 0.2));
  let remainder = from_scales(alpha, c, (2.0, 6.0), 1.0);
  let eps = 1e-8_f64;
  let (_, variance) = remainder.rates(eps.powf(-alpha) / remainder.rate);
  let stable = (c.0 + c.1) * eps.powf(2.0 - alpha) / (2.0 - alpha);
  close(variance, stable, 1e-6, "variance at ε = 1e-8");
}

/// With no arrival the remainder is the whole law: variance `Σ C Γ(2 − α) λ^{α−2}`, mean
/// `Σ ± C Γ(1 − α) λ^{α−1}`.
#[test]
fn no_arrival_leaves_the_whole_law() {
  use stochastic_rs_distributions::special::gamma;
  let (alpha, c, lambda) = (1.5f64, 0.3f64, (2.0f64, 6.0f64));
  let (drift, variance) = from_scales(alpha, (c, c), lambda, 1.0).rates(0.0);
  let whole = c * gamma(2.0 - alpha) * (lambda.0.powf(alpha - 2.0) + lambda.1.powf(alpha - 2.0));
  let mean = c * gamma(1.0 - alpha) * (lambda.0.powf(alpha - 1.0) - lambda.1.powf(alpha - 1.0));
  close(variance, whole, 1e-10, "variance with no arrival");
  close(drift, mean, 1e-9, "drift with no arrival");
}

/// The Gamma(`terms`, 1) average of `f` by a 20 000-point midpoint rule, apart from the
/// eight-point Gauss–Legendre panels under test.
fn over_last_arrival(terms: usize, f: impl Fn(f64) -> f64) -> f64 {
  use stochastic_rs_distributions::special::ln_gamma;
  let j = terms as f64;
  let (lo, hi) = ((j - 12.0 * j.sqrt()).max(0.0), j + 12.0 * j.sqrt() + 60.0);
  let steps = 20_000;
  let h = (hi - lo) / steps as f64;
  let (mut mass, mut sum) = (0.0, 0.0);
  for i in 0..steps {
    let g = lo + (i as f64 + 0.5) * h;
    let w = ((j - 1.0) * g.ln() - g - ln_gamma(j)).exp();
    mass += w;
    sum += w * f(g);
  }
  sum / mass
}

/// The launch drift is the path drift averaged over `Γ_J ~ Gamma(J, 1)`; with equal sides and
/// tempering the kept terms have no mean and the launch variance is that average too.
#[test]
fn the_expected_rates_average_over_the_last_arrival() {
  for terms in [1_usize, 3, 40, 4_000] {
    let skewed = from_scales(1.5, (0.5, 0.2), (2.0, 6.0), 1.0);
    let drift = over_last_arrival(terms, |g| skewed.rates(g).0);
    close(
      skewed.expected_rates(terms).0,
      drift,
      2e-4,
      &format!("drift at J = {terms}"),
    );
    let even = from_scales(1.5, (0.3, 0.3), (4.0, 4.0), 1.0);
    let variance = over_last_arrival(terms, |g| even.rates(g).1);
    close(
      even.expected_rates(terms).1,
      variance,
      1e-5,
      &format!("variance at J = {terms}"),
    );
  }
}

/// One term: `Var H(Γ_1) = E[h₂] − E[h₁]²` with `h₁ = −T μ′`, `h₂ = −T V′` the term's moments at
/// its arrival, so the launch variance must be the law's less that.
#[test]
fn the_expected_variance_completes_one_kept_term_to_the_law() {
  let remainder = from_scales(1.5, (0.5, 0.2), (2.0, 6.0), 1.0);
  let slope = |g: f64, pick: fn((f64, f64)) -> f64| {
    let h = 1e-4 * g;
    -(pick(remainder.rates(g + h)) - pick(remainder.rates(g - h))) / (2.0 * h)
  };
  let first = over_last_arrival(1, |g| slope(g, |r| r.0));
  let second = over_last_arrival(1, |g| slope(g, |r| r.1));
  let whole = remainder.rates(0.0).1;
  let (_, variance) = remainder.expected_rates(1);
  close(
    variance,
    whole - (second - first * first),
    1e-4,
    "launch variance at J = 1",
  );
}

/// KoBoL with unequal sides, `α = 1.5`: the law's variance less the kept terms', 400 000 paths
/// in numpy (`evidence/device-gap.log`): 0.5538 ± 0.0013 at `J = 64`, 0.3783 ± 0.0017 at 256.
#[test]
fn the_expected_variance_completes_the_kept_terms_of_unequal_sides() {
  let remainder = SeriesRemainder::new(1.5, 1.5 / 1.2, 1.0, 2.0 / 3.0, 3.0, 5.0, 1.0, 1.0);
  for (terms, want, se) in [(64_usize, 0.5538, 0.0013), (256, 0.3783, 0.0017)] {
    let (_, variance) = remainder.expected_rates(terms);
    assert!(
      (variance - want).abs() < 5.0 * se,
      "J = {terms}: launch variance {variance}, Monte Carlo {want} ± {se}"
    );
  }
}

#[test]
fn a_cut_too_small_to_represent_leaves_nothing() {
  let remainder = from_scales(0.01, (0.3, 0.3), (2.0, 6.0), 1.0);
  assert_eq!(remainder.rates(1e6), (0.0, 0.0));
}
