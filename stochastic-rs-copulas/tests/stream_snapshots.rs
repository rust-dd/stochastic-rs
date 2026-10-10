//! Seeded copula streams pinned by hash; a row moves only with a `### Numerical changes` entry
//! (`STREAM_SNAPSHOT_REGEN=1 cargo test -p stochastic-rs-copulas --test stream_snapshots -- --nocapture` prints the table).

use ndarray::Array1;
use ndarray::Array2;
use ndarray::array;
use stochastic_rs_copulas::bivariate::amh::Amh;
use stochastic_rs_copulas::bivariate::bb1::Bb1;
use stochastic_rs_copulas::bivariate::bb7::Bb7;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::fgm::Fgm;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::galambos::Galambos;
use stochastic_rs_copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs_copulas::bivariate::gumbel::Gumbel;
use stochastic_rs_copulas::bivariate::husler_reiss::HuslerReiss;
use stochastic_rs_copulas::bivariate::joe::Joe;
use stochastic_rs_copulas::bivariate::plackett::Plackett;
use stochastic_rs_copulas::bivariate::t_copula::TCopula;
use stochastic_rs_copulas::empirical::EmpiricalCopula2D;
use stochastic_rs_copulas::multivariate::cvine::CVine;
use stochastic_rs_copulas::multivariate::dvine::DVine;
use stochastic_rs_copulas::multivariate::dvine::PairCopula;
use stochastic_rs_copulas::multivariate::gaussian::GaussianMultivariate;
use stochastic_rs_copulas::multivariate::nac::NacFamily;
use stochastic_rs_copulas::multivariate::nac::NacNode;
use stochastic_rs_copulas::multivariate::nac::NestedArchimedean;
use stochastic_rs_copulas::multivariate::rvine::RVine;
use stochastic_rs_copulas::multivariate::t::TMultivariate;
use stochastic_rs_copulas::multivariate::tree::TreeMultivariate;
use stochastic_rs_copulas::multivariate::vine::VineMultivariate;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;

const SEED: u64 = 7;
const N: usize = 64;

/// `(name, fnv1a of the bits of the `(N, d)` sample)`.
const PINS: &[(&str, u64)] = &[
  ("amh", 0x589e3274eaaa51e8),
  ("bb1", 0x08f7414f5c289cb7),
  ("bb7", 0x628f49f39de1c0e2),
  ("clayton", 0xde46259e4c14fee6),
  ("fgm", 0x1b30ac6f627924fb),
  ("frank", 0x2f9cd7cf7868e762),
  ("galambos", 0x0085c28ef4263794),
  ("gaussian", 0xa328de38b1bfb31d),
  ("gumbel", 0x6cebaa664f917946),
  ("husler_reiss", 0x8060fcf627b39fba),
  ("joe", 0x062c78648bd84254),
  ("plackett", 0x1412f43d3850c3f0),
  ("t_copula", 0xf7d437c24da6ea30),
  ("gaussian_multivariate", 0xa9e6f7560842916c),
  ("t_multivariate", 0x6bcffcbcf2ecf0d0),
  ("vine_multivariate", 0xa9e6f7560842916c),
  ("tree_multivariate", 0xa9e6f7560842916c),
  ("cvine", 0xad38b89d85dc8c41),
  ("dvine", 0xfc183eb80e04637d),
  ("rvine_independence", 0x7fc1a032d2ef1540),
  ("nac_gumbel_nested", 0x2a6a150ab6e65bf6),
  ("nac_clayton_flat", 0xc0bf2c344f128602),
  ("empirical", 0x3e1dc30dfbde113d),
];

fn fnv1a(values: impl Iterator<Item = u64>) -> u64 {
  let mut hash = 0xcbf2_9ce4_8422_2325_u64;
  for value in values {
    for byte in value.to_le_bytes() {
      hash ^= u64::from(byte);
      hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
  }
  hash
}

fn hash(sample: &Array2<f64>) -> u64 {
  fnv1a(sample.iter().map(|x| x.to_bits()))
}

fn with_theta<C: BivariateExt>(mut copula: C, theta: f64) -> C {
  copula.set_theta(theta);
  copula
}

fn corr3() -> Array2<f64> {
  array![[1.0, 0.7, 0.3], [0.7, 1.0, 0.5], [0.3, 0.5, 1.0]]
}

fn pair_trees() -> Vec<Vec<PairCopula>> {
  vec![
    vec![
      PairCopula::Gaussian { rho: 0.6 },
      PairCopula::Clayton { theta: 2.0 },
    ],
    vec![PairCopula::Independence],
  ]
}

fn empirical() -> EmpiricalCopula2D {
  let x = Array1::from_vec((0..50).map(|i| i as f64).collect());
  let y = Array1::from_vec((0..50).map(|i| (i as f64 * 1.7) % 37.0).collect());
  EmpiricalCopula2D::new_from_two_series(&x, &y)
}

fn samples() -> Vec<(&'static str, Array2<f64>)> {
  let mut rows: Vec<(&'static str, Array2<f64>)> = Vec::new();
  macro_rules! row {
    ($name:literal, $sample:expr) => {
      rows.push(($name, $sample));
    };
  }
  row!(
    "amh",
    with_theta(Amh::new(), 0.5)
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "bb1",
    Bb1::new(Some(0.8), Some(1.6), None)
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "bb7",
    Bb7::new(Some(1.7), Some(0.9), None)
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "clayton",
    with_theta(Clayton::new(), 2.0)
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "fgm",
    with_theta(Fgm::new(), 0.5)
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "frank",
    Frank::new(Some(4.0), None)
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "galambos",
    with_theta(Galambos::new(), 1.5)
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "gaussian",
    with_theta(GaussianCopula::new(), 0.5)
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "gumbel",
    Gumbel::new(Some(2.0), None)
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "husler_reiss",
    with_theta(HuslerReiss::new(), 1.5)
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "joe",
    with_theta(Joe::new(), 2.0)
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "plackett",
    with_theta(Plackett::new(), 3.0)
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "t_copula",
    with_theta(TCopula::with_nu(4.0), 0.5)
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "gaussian_multivariate",
    GaussianMultivariate::new_with_corr(corr3())
      .unwrap()
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "t_multivariate",
    TMultivariate::new_with(corr3(), 5.0)
      .unwrap()
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "vine_multivariate",
    VineMultivariate::new_with_corr(corr3())
      .unwrap()
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "tree_multivariate",
    TreeMultivariate::new_with_corr(corr3())
      .unwrap()
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "cvine",
    CVine::new(3, pair_trees())
      .unwrap()
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "dvine",
    DVine::new(3, pair_trees())
      .unwrap()
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!(
    "rvine_independence",
    RVine::independence(3)
      .unwrap()
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  let nested = NacNode {
    theta: 1.5,
    leaves: vec![2],
    children: vec![NacNode::leaf_group(3.0, vec![0, 1])],
  };
  row!(
    "nac_gumbel_nested",
    NestedArchimedean::new(NacFamily::Gumbel, nested, 3)
      .unwrap()
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  let flat = NacNode::leaf_group(2.0, vec![0, 1, 2]);
  row!(
    "nac_clayton_flat",
    NestedArchimedean::new(NacFamily::Clayton, flat, 3)
      .unwrap()
      .sample_with_seed(N, SEED)
      .unwrap()
  );
  row!("empirical", empirical().sample_with_seed(N, SEED));
  rows
}

#[test]
fn seeded_copula_streams_are_pinned() {
  let rows = samples();
  if std::env::var_os("STREAM_SNAPSHOT_REGEN").is_some() {
    for (name, sample) in &rows {
      println!("  (\"{name}\", 0x{:016x}),", hash(sample));
    }
    return;
  }
  assert_eq!(
    rows.len(),
    PINS.len(),
    "{} samplers, {} pins",
    rows.len(),
    PINS.len()
  );
  for ((name, sample), (pinned, want)) in rows.iter().zip(PINS) {
    assert_eq!(name, pinned);
    assert_eq!(hash(sample), *want, "{name} moved");
  }
}

#[test]
fn the_same_seed_replays_every_sampler() {
  let first = samples();
  let second = samples();
  for ((name, a), (_, b)) in first.iter().zip(&second) {
    assert_eq!(a, b, "{name} is not a function of its seed");
  }
}
