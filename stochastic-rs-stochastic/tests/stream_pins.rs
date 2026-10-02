use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::interest::duffie_kan_jump_exp::DuffieKanJumpExp;
use stochastic_rs_stochastic::jump::mjd_log::MjdLog;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::process::subordinator::poisson_subordinator::PoissonSubordinator;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::bates_svj::BatesSvj;
use stochastic_rs_stochastic::volatility::fbates_svj::FBatesSvj;
use stochastic_rs_stochastic::volatility::hkde::Hkde;

const N: usize = 24;
const SEED: u64 = 42;
const LAMBDA: f64 = 20.0;

fn pin(label: &str, got: &Array1<f64>, golden: [u64; 8]) {
  let bits = got.iter().take(8).map(|x| x.to_bits()).collect::<Vec<_>>();
  assert_eq!(bits.len(), 8, "{label}: the path is shorter than the pin");
  for (i, (&a, &gb)) in got.iter().zip(&golden).enumerate() {
    let g = f64::from_bits(gb);
    assert!(
      (a - g).abs() <= 1e-9 * (1.0 + g.abs()),
      "{label}[{i}]: got {a:?}, pinned {g:?}; captured bits {bits:?}"
    );
  }
}

#[test]
fn poisson_horizon_mode_is_pinned() {
  let path =
    Poisson::<f64, _>::new(2.0 * LAMBDA, None, Some(1.0), Deterministic::new(SEED)).sample();
  pin(
    "Poisson horizon",
    &path,
    [
      0,
      4580482050277747696,
      4586085390756737535,
      4590436827944522992,
      4592312919580819814,
      4594241694543480592,
      4594793799711395387,
      4595579018267019823,
    ],
  );
}

#[test]
fn mjd_log_is_pinned() {
  let path = MjdLog::new(
    Some(0.05),
    None,
    None,
    None,
    0.2,
    LAMBDA,
    0.0,
    0.1,
    N,
    Some(100.0),
    Some(1.0),
    Deterministic::new(SEED),
  )
  .sample();
  pin(
    "MjdLog",
    &path,
    [
      4636737291354636288,
      4635808779003138008,
      4636717111095872502,
      4636257093357992921,
      4636746937467317719,
      4635851941351680734,
      4635437851904418731,
      4635525806424921260,
    ],
  );
}

#[test]
fn bates_svj_is_pinned() {
  let [s, _] = BatesSvj::new(
    Some(0.05),
    None,
    None,
    None,
    LAMBDA,
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
    Deterministic::new(SEED),
  )
  .sample();
  pin(
    "BatesSvj",
    &s,
    [
      4636737291354636288,
      4637504339693668260,
      4636306926157148593,
      4636825074432907165,
      4634842882100959005,
      4634841034470317663,
      4635450031313721238,
      4636081128162138464,
    ],
  );
}

#[test]
fn f_bates_svj_is_pinned() {
  let [s, _] = FBatesSvj::new(
    0.1,
    0.05,
    100.0,
    0.04,
    0.04,
    2.0,
    0.3,
    -0.7,
    LAMBDA,
    -0.01,
    0.1,
    N,
    Some(1.0),
    Deterministic::new(SEED),
  )
  .sample();
  pin(
    "FBatesSvj",
    &s,
    [
      4636737291354636288,
      4636488598398058643,
      4636391219181329117,
      4636372562052597747,
      4636434497921378406,
      4636410522307895968,
      4636069901416030000,
      4636959531447873207,
    ],
  );
}

#[test]
fn hkde_is_pinned() {
  let [s, _] = Hkde::new(
    0.05,
    1.5,
    0.04,
    0.3,
    -0.7,
    0.04,
    LAMBDA,
    0.4,
    5.0,
    5.0,
    N,
    Some(100.0),
    Some(1.0),
    Some(false),
    Deterministic::new(SEED),
  )
  .sample();
  pin(
    "Hkde",
    &s,
    [
      4636737291354636288,
      4636084769398645750,
      4634218419775452295,
      4633955566856843868,
      4633352557550870126,
      4633447302741357346,
      4633857874403438776,
      4634668288891836423,
    ],
  );
}

#[test]
fn poisson_subordinator_is_pinned() {
  let path =
    PoissonSubordinator::new(LAMBDA, N, Some(0.0), Some(1.0), Deterministic::new(SEED)).sample();
  pin(
    "PoissonSubordinator",
    &path,
    [
      0,
      4607182418800017408,
      4611686018427387904,
      4611686018427387904,
      4616189618054758400,
      4616189618054758400,
      4617315517961601024,
      4618441417868443648,
    ],
  );
}

#[test]
fn duffie_kan_jump_exp_is_pinned() {
  let [_, x] = DuffieKanJumpExp::new(
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
    LAMBDA,
    0.05,
    N,
    Some(0.05),
    Some(0.05),
    Some(1.0),
    Deterministic::new(SEED),
  )
  .sample();
  pin(
    "DuffieKanJumpExp",
    &x,
    [
      4587366580439587226,
      4587460228945520871,
      13813459079803119252,
      13817196518208930752,
      13817292017596882057,
      13817240560671561989,
      13807076140597773348,
      13807229549475133900,
    ],
  );
}
