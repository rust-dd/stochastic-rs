//! Recorded sheet bits, chunked streams, and owned output at the sampling boundary.

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::sync::{Arc, Barrier};

use stochastic_rs_core::simd_rng::{Deterministic, SeedExt};
use stochastic_rs_stochastic::{
  device::{DeviceError, Metal, SheetBackend},
  sheet::fbs::Fbs,
  traits::ProcessExt,
};

const GRIDS: [(usize, usize); 14] = [
  (2, 2),
  (3, 5),
  (5, 3),
  (9, 17),
  (17, 9),
  (33, 65),
  (129, 17),
  (17, 129),
  (257, 65),
  (3, 1025),
  (1025, 3),
  (5, 2049),
  (2049, 5),
  (3, 4097),
];

fn process(m: usize, n: usize, h: f32, r: f32, seed: u64) -> Fbs<f32, Deterministic, Metal> {
  Fbs::new(h, m, n, r, Deterministic::new(seed)).on::<Metal>()
}

#[test]
fn seeded_sheets_match_the_recording() {
  let mut pins = include_str!("fixtures/metal_sheet.txt").lines();
  for (m, n) in GRIDS {
    for (h, r) in [(0.3, 1.0), (0.7, 1.0), (0.9, 2.0)] {
      let sheets = process(m, n, h, r, 37).sample_par(3);
      for (sheet, values) in sheets.iter().enumerate() {
        for i in [0, m / 2, m - 1] {
          for j in [0, n / 2, n - 1] {
            let fields = pins.next().unwrap().split_whitespace().collect::<Vec<_>>();
            assert_eq!(fields[0].parse::<usize>().unwrap(), m);
            assert_eq!(fields[1].parse::<usize>().unwrap(), n);
            assert_eq!(fields[2].parse::<f32>().unwrap(), h);
            assert_eq!(fields[3].parse::<f32>().unwrap(), r);
            assert_eq!(fields[4].parse::<usize>().unwrap(), sheet);
            assert_eq!(fields[5].parse::<usize>().unwrap(), i);
            assert_eq!(fields[6].parse::<usize>().unwrap(), j);
            let want = fields[7].parse::<u32>().unwrap();
            let got = values[(i, j)].to_bits();
            assert_eq!(
              got, want,
              "{m}x{n}, h={h}, r={r}, sheet={sheet}, ({i},{j}): {got:08x} != {want:08x}"
            );
          }
        }
      }
    }
  }
  assert!(pins.next().is_none());
}

#[test]
fn rectangular_sheets_keep_their_stream_across_chunk_boundaries() {
  for (m, n) in [
    (2, 2),
    (3, 17),
    (17, 3),
    (9, 17),
    (65, 33),
    (3, 2049),
    (2049, 3),
  ] {
    let make = || process(m, n, 0.7, 1.0, 41);
    let whole = make().try_sample_par(7).unwrap();
    let bytes = (4 * 4 * (m - 1) * (n - 1) + m * n) * 4;
    for chunk in [1, 2, 3, 5] {
      let chunked = make()
        .with_backend(Metal::default().with_batch_budget(bytes * chunk))
        .try_sample_par(7)
        .unwrap();
      assert_eq!(chunked, whole, "{m}x{n}, chunk={chunk}");
    }
    assert_ne!(whole[0], whole[1]);
  }
}

#[test]
fn public_sampling_forms_and_empty_batches_advance_one_seed() {
  let make = || process(17, 9, 0.7, 1.0, 43);
  assert_eq!(make().sample(), make().sample_par(1)[0]);
  assert_eq!(make().try_sample().unwrap(), make().sample());
  assert_eq!(make().try_sample_par(5).unwrap(), make().sample_par(5));
  assert_eq!(make().sample_map(5, Clone::clone), make().sample_par(5));
  let fbs = make();
  let expected = fbs.seed.clone();
  for sheets in [0, 1, 0, 5] {
    assert_eq!(fbs.try_sample_par(sheets).unwrap().len(), sheets);
    expected.next_seed();
    assert_eq!(fbs.seed.current(), expected.current());
  }
  let invalid = make().with_backend(Metal::new(usize::MAX));
  let expected = invalid.seed.clone();
  assert!(matches!(
    invalid.try_sample(),
    Err(DeviceError::Unavailable(_))
  ));
  expected.next_seed();
  assert_eq!(invalid.seed.current(), expected.current());
  assert!(invalid.try_sample_par(0).unwrap().is_empty());
}

#[test]
fn direct_sheet_calls_reject_unsupported_dimensions_before_opening_a_device() {
  let overflowing = (1_usize << (usize::BITS - 1)) + 1;
  for (m, n) in [(0, 5), (1, 5), (4, 5), (9, 4), (overflowing, 5)] {
    let mut fbs = process(9, 5, 0.7, 1.0, 61).with_backend(Metal::new(usize::MAX));
    fbs.m = m;
    fbs.n = n;
    let expected = fbs.seed.clone();
    assert!(matches!(
      fbs.backend.try_sheet(&fbs),
      Err(DeviceError::Launch(_))
    ));
    expected.next_seed();
    assert_eq!(fbs.seed.current(), expected.current());
    assert!(fbs.backend.try_sheets(&fbs, 0).unwrap().is_empty());
    expected.next_seed();
    assert_eq!(fbs.seed.current(), expected.current());
  }
}

#[test]
fn direct_sheet_calls_reject_a_stale_embedding_shape() {
  for (m, n) in [(17, 5), (5, 9)] {
    let mut fbs = process(9, 5, 0.7, 1.0, 67).with_backend(Metal::new(usize::MAX));
    fbs.m = m;
    fbs.n = n;
    assert!(matches!(
      fbs.backend.try_sheets(&fbs, 3),
      Err(DeviceError::Launch(_))
    ));
    assert!(fbs.backend.try_sheets(&fbs, 0).unwrap().is_empty());
  }
}

#[test]
fn mutated_parameters_do_not_reuse_another_process_embedding() {
  for (stored_hurst, stored_r) in [(0.3, 1.0), (0.7, 2.0)] {
    let make = move |mutated| {
      let mut fbs = if mutated {
        process(17, 9, stored_hurst, stored_r, 71)
      } else {
        process(17, 9, 0.7, 1.0, 71)
      };
      fbs.hurst = 0.7;
      fbs.r = 1.0;
      fbs
    };
    let expected = [false, true].map(|mutated| {
      std::thread::spawn(move || make(mutated).sample_par(3))
        .join()
        .unwrap()
    });
    assert_ne!(expected[0], expected[1]);
    for order in [[false, true], [true, false]] {
      let expected = expected.clone();
      std::thread::spawn(move || {
        for mutated in [order[0], order[1], order[0], order[1]] {
          assert_eq!(make(mutated).sample_par(3), expected[usize::from(mutated)]);
        }
      })
      .join()
      .unwrap();
    }
  }
}

#[test]
fn returned_sheets_survive_concurrent_cache_reuse() {
  let barrier = Arc::new(Barrier::new(3));
  let workers = (0..3)
    .map(|worker| {
      let barrier = barrier.clone();
      std::thread::spawn(move || {
        let make = || process(17, 9, 0.7, 1.0, 47 + worker);
        let expected = make().try_sample_par(3);
        barrier.wait();
        let expected = expected.unwrap();
        for round in 0..5 {
          let actual = make().sample_map(3, |sheet| {
            let before = sheet.clone();
            let other = process(9, 17, 0.3, 1.0, 53 + round).sample();
            assert!(other.iter().all(|value| value.is_finite()));
            assert_eq!(sheet, &before);
            before
          });
          assert_eq!(actual, expected);
        }
      })
    })
    .collect::<Vec<_>>();
  for worker in workers {
    worker.join().unwrap();
  }
}

fn standardized_error(values: impl Iterator<Item = f64>, expected: f64) -> f64 {
  let (mut count, mut mean, mut squared) = (0.0, 0.0, 0.0);
  for value in values {
    assert!(value.is_finite());
    count += 1.0;
    let delta = value - mean;
    mean += delta / count;
    squared += delta * (value - mean);
  }
  let error = (mean - expected).abs() / (squared / (count - 1.0) / count).sqrt();
  assert!(error.is_finite());
  error
}

/// The increment covariance is |s|^(2H) + |t|^(2H) - |s-t|^(2H).
/// Kroese & Botev (2015), Spatial process simulation, §4.4, arXiv:1308.0399.
#[test]
fn rectangular_field_increments_have_the_isotropic_covariance() {
  for h in [0.3_f32, 0.7] {
    let exponent = 2.0 * f64::from(h);
    let y = 4.0 / 17.0_f64;
    let x = 16.0 / 33.0_f64;
    let expected = [
      2.0 * y.powf(exponent),
      2.0 * x.powf(exponent),
      y.powf(exponent) + x.powf(exponent) - y.hypot(x).powf(exponent),
    ];
    let best = [311, 313, 317]
      .into_iter()
      .map(|seed| {
        let sheets = process(17, 33, h, 1.0, seed).sample_par(4096);
        let increments = ndarray::Array2::from_shape_fn((sheets.len(), 2), |(i, axis)| {
          let point = if axis == 0 { (4, 0) } else { (0, 16) };
          f64::from(sheets[i][point] - sheets[i][(0, 0)])
        });
        let mut worst = 0.0_f64;
        for (statistic, want) in expected.into_iter().enumerate() {
          let values = increments.outer_iter().map(|v| match statistic {
            0 => v[0] * v[0],
            1 => v[1] * v[1],
            _ => v[0] * v[1],
          });
          worst = worst.max(standardized_error(values, want));
        }
        let independent =
          (0..sheets.len() / 2).map(|i| increments[(2 * i, 0)] * increments[(2 * i + 1, 0)]);
        worst.max(standardized_error(independent, 0.0))
      })
      .fold(f64::INFINITY, f64::min);
    assert!(
      best < 5.0,
      "h={h}: covariance differs by {best} standard errors"
    );
  }
}
