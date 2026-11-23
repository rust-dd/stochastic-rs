//! Monte Carlo of the rough and the lifted Heston models against their characteristic functions.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::gamma;
use stochastic_rs_quant::pricing::fourier::RoughHeston;
use stochastic_rs_quant::pricing::fourier::RoughHestonCf;
use stochastic_rs_quant::pricing::fourier::RoughHestonParams;
use stochastic_rs_quant::traits::ModelPricer;
use stochastic_rs_stochastic::rough::RlHeston;
use stochastic_rs_stochastic::traits::ProcessExt;

const STRIKES: [f64; 3] = [0.9, 1.0, 1.1];
const TAU: f64 = 0.5;
const PATHS: usize = 40_000;

/// `ν = 0.15` keeps the variance off zero (0.2 % of `RlHeston`'s steps); at 0.3 it sits there 9 %
/// of them and both simulators' floors price every call high.
const P: RoughHestonParams = RoughHestonParams {
  v0: 0.04,
  kappa: 1.0,
  theta: 0.04,
  nu: 0.15,
  rho: -0.7,
  hurst: 0.1,
};

/// Each strike's mean call payoff and its standard error.
fn estimates(terminal: &[f64]) -> [(f64, f64); 3] {
  let m = terminal.len() as f64;
  STRIKES.map(|k| {
    let payoff = |s: &f64| (s - k).max(0.0);
    let mean = terminal.iter().map(payoff).sum::<f64>() / m;
    let square = terminal
      .iter()
      .map(|s| (payoff(s) - mean).powi(2))
      .sum::<f64>();
    (mean, (square / (m - 1.0) / m).sqrt())
  })
}

/// `|MC − chf price| < 4·se + bias` at each strike, `bias` the simulator's stated allowance.
fn assert_within_error_bars(what: &str, terminal: &[f64], model: &RoughHeston, bias: f64) {
  for ((mean, se), k) in estimates(terminal).into_iter().zip(STRIKES) {
    let want = model.price_call(1.0, k, 0.0, 0.0, TAU);
    let bound = 4.0 * se + bias;
    assert!(
      (mean - want).abs() < bound,
      "{what} at {k}: {mean} ± {se} vs {want}"
    );
  }
}

/// `RlHeston`'s paths against the Adams solution. Its Euler step prices calls high: by 4.6e-4 to
/// 5.6e-4 at these strikes on 257 points and 160 000 paths, and by half that on 1025 points, the
/// rate `Δ^{H+½}`; 6e-4 is that allowance, beside four standard errors at a fixed seed.
#[test]
fn rl_heston_paths_price_as_the_adams_characteristic_function() {
  const POINTS: usize = 257;
  const EULER_BIAS: f64 = 6e-4;
  let model = RoughHeston::new(P, RoughHestonCf::Adams { steps: 100 });
  let seed = Deterministic::new(20_261_123);
  let process = RlHeston::new(
    P.hurst,
    1.0,
    P.v0,
    P.kappa,
    P.theta,
    P.nu,
    P.rho,
    0.0,
    POINTS,
    Some(TAU),
    None,
    seed,
  );
  let terminal = process.sample_map(2 * PATHS, |[s, _]| s[POINTS - 1]);
  assert_within_error_bars("RlHeston", &terminal, &model, EULER_BIAS);
  let (mean, _) = estimates(&terminal)[1];
  let exact = model.price_call(1.0, 1.0, 0.0, 0.0, TAU);
  assert!(
    mean > exact,
    "the allowance is one-sided: {mean} vs {exact}"
  );
}

/// Abi Jaber's (3.3) and his scheme (C.3)–(C.4) with a log-Euler spot, one path's terminal spot.
struct LiftedScheme {
  weights: Vec<f64>,
  reversions: Vec<f64>,
  steps: usize,
}

impl LiftedScheme {
  fn new(factors: usize, ratio: f64, steps: usize) -> Self {
    let alpha = P.hurst + 0.5;
    let half = 0.5 * factors as f64;
    let step = ratio.powf(1.0 - alpha) - 1.0;
    let scale =
      step * ratio.powf((alpha - 1.0) * (1.0 + half)) / (gamma(alpha) * gamma(2.0 - alpha));
    let slope = (1.0 - alpha) / (2.0 - alpha) * (ratio.powf(2.0 - alpha) - 1.0) / step;
    let index = |i: usize| (i + 1) as f64;
    Self {
      weights: (0..factors)
        .map(|i| scale * ratio.powf((1.0 - alpha) * index(i)))
        .collect(),
      reversions: (0..factors)
        .map(|i| slope * ratio.powf(index(i) - 1.0 - half))
        .collect(),
      steps,
    }
  }

  /// `g₀(t) = V₀ + κθΣcᵢ(1 − e^{−xᵢt})/xᵢ`, his (2.6).
  fn input_curve(&self, t: f64) -> f64 {
    let factors = self.weights.iter().zip(&self.reversions);
    P.v0
      + P.kappa
        * P.theta
        * factors
          .map(|(c, x)| c * -(-x * t).exp_m1() / x)
          .sum::<f64>()
  }

  fn terminal(&self, normals: &[f64]) -> f64 {
    let dt = TAU / self.steps as f64;
    let mut u = vec![0.0; self.weights.len()];
    let mut log_spot = 0.0;
    for (k, pair) in normals.as_chunks::<2>().0.iter().enumerate() {
      let factors = self.weights.iter().zip(&u);
      let variance = self.input_curve(dt * k as f64) + factors.map(|(c, u)| c * u).sum::<f64>();
      let floored = variance.max(0.0);
      let dw = dt.sqrt() * pair[0];
      let db = P.rho * dw + (1.0 - P.rho * P.rho).sqrt() * dt.sqrt() * pair[1];
      log_spot += -0.5 * floored * dt + floored.sqrt() * db;
      for (u, x) in u.iter_mut().zip(&self.reversions) {
        *u = (*u - P.kappa * variance * dt + P.nu * floored.sqrt() * dw) / (1.0 + x * dt);
      }
    }
    log_spot.exp()
  }
}

/// The lifted model simulated by its author's scheme against its characteristic function, four
/// standard errors at a fixed seed; 240 000 paths on 400 steps sit 0.8 to 1.3 of theirs away.
/// Reference: Abi Jaber (2019), Quant. Finance 19(12), 1995–2013, DOI 10.1080/14697688.2019.1615113
#[test]
fn the_lifted_scheme_s_paths_price_as_the_lifted_characteristic_function() {
  let model = RoughHeston::new(P, RoughHestonCf::LIFTED);
  const STEPS: usize = 200;
  let scheme = LiftedScheme::new(20, 2.5, STEPS);
  let mut stream = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(47));
  let mut normals = vec![0.0; 2 * STEPS];
  let terminal = (0..PATHS / 2).map(|_| {
    stream.fill_slice(&mut normals);
    scheme.terminal(&normals)
  });
  let terminal = terminal.collect::<Vec<_>>();
  assert_within_error_bars("lifted scheme", &terminal, &model, 0.0);
}
