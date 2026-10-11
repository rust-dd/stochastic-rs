//! Recorded seeded Metal bits and concurrent ownership at the sampling boundary.

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

use stochastic_rs_core::simd_rng::{Deterministic, SeedExt};
use stochastic_rs_stochastic::{
  device::{DeviceError, Metal},
  noise::fgn::Fgn,
  traits::ProcessExt,
};

#[test]
fn zero_step_samples_preserve_the_requested_row_count() {
  for budget in [8, usize::MAX] {
    let process = Fgn::new(0.7_f32, 0, Some(1.0), Deterministic::new(37))
      .with_backend(Metal::default().with_batch_budget(budget));
    let expected_seed = process.seed().clone();
    assert!(process.try_sample().unwrap().is_empty());
    expected_seed.next_seed();
    assert_eq!(process.seed().current(), expected_seed.current());
    for count in [0, 1, 5] {
      let rows = process.try_sample_par(count).unwrap();
      assert_eq!(rows.len(), count);
      assert!(rows.iter().all(|row| row.is_empty()));
      expected_seed.next_seed();
      assert_eq!(process.seed().current(), expected_seed.current());

      let calls = AtomicUsize::new(0);
      let lengths = process.sample_map(count, |row| {
        calls.fetch_add(1, Ordering::Relaxed);
        row.len()
      });
      assert_eq!(lengths, vec![0; count]);
      assert_eq!(calls.load(Ordering::Relaxed), count);
      expected_seed.next_seed();
      assert_eq!(process.seed().current(), expected_seed.current());

      let calls = AtomicUsize::new(0);
      let lengths = process.sample_map_view(count, |row| {
        calls.fetch_add(1, Ordering::Relaxed);
        row.len()
      });
      assert_eq!(lengths, vec![0; count]);
      assert_eq!(calls.load(Ordering::Relaxed), count);
      expected_seed.next_seed();
      assert_eq!(process.seed().current(), expected_seed.current());
    }
  }
  let unavailable =
    Fgn::new(0.7_f32, 0, Some(1.0), Deterministic::new(37)).with_backend(Metal::new(usize::MAX));
  let expected_seed = unavailable.seed().clone();
  assert!(matches!(
    unavailable.try_sample(),
    Err(DeviceError::Unavailable(_))
  ));
  expected_seed.next_seed();
  assert_eq!(unavailable.seed().current(), expected_seed.current());
}

#[test]
fn seeded_rows_match_the_recording() {
  let mut pins = include_str!("fixtures/metal_fgn.txt").lines();
  for n in [1, 2, 8, 100, 1024, 2048, 3000, 4096, 16384, 65536] {
    for h in [0.3_f32, 0.5, 0.7] {
      let process = Fgn::new(h, n, Some(2.5), Deterministic::new(37)).on::<Metal>();
      let rows = process.sample_par(3);
      for (r, row) in rows.iter().enumerate() {
        for j in [0, n / 2, n - 1] {
          let fields = pins.next().unwrap().split_whitespace().collect::<Vec<_>>();
          assert_eq!(fields[0].parse::<usize>().unwrap(), n);
          assert_eq!(fields[1].parse::<f32>().unwrap(), h);
          assert_eq!(fields[2].parse::<usize>().unwrap(), r);
          assert_eq!(fields[3].parse::<usize>().unwrap(), j);
          let want = fields[4].parse::<u32>().unwrap();
          let got = row[j].to_bits();
          assert_eq!(
            got, want,
            "n={n}, h={h}, row={r}, j={j}: {got:08x} != {want:08x}"
          );
        }
      }
    }
  }
  assert!(pins.next().is_none());
}

#[test]
fn concurrent_borrowed_rows_remain_owned_until_consumed() {
  let barrier = Arc::new(Barrier::new(3));
  let handles = (0..3)
    .map(|worker| {
      let barrier = barrier.clone();
      std::thread::spawn(move || {
        let make =
          || Fgn::new(0.7_f32, 4096, Some(1.0), Deterministic::new(40 + worker)).on::<Metal>();
        let expected = make().try_sample_par(3);
        barrier.wait();
        let expected = expected.unwrap();
        for _ in 0..5 {
          let actual = make().sample_map_view(3, |row| {
            let before = row.to_owned();
            std::thread::sleep(std::time::Duration::from_millis(1));
            assert_eq!(row, before.view());
            before
          });
          assert_eq!(actual, expected);
        }
      })
    })
    .collect::<Vec<_>>();
  for handle in handles {
    handle.join().unwrap();
  }
}
