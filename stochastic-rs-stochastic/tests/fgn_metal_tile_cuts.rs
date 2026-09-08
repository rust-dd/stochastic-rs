//! Every cut the Metal fGN pipeline can take, held against the host's own
//! sampler and against the covariance the circulant embedding reproduces.
//!
//! The pipeline runs the first `k` butterfly stages inside one threadgroup's
//! tile and whatever is left as its own dispatch, with the last stage always
//! outside because it exports the read-out. Where `k` falls depends on the
//! transform length and on how wide a threadgroup the compiled kernel admits,
//! so a short transform is one tile pass and the read-out while a long one
//! keeps stages between them — two different index paths through the same
//! butterflies. An index that is wrong in either still produces plausible
//! Gaussian numbers; what it destroys is the autocovariance.
//!
//! The device and the host draw different streams, so what is pinned is the
//! law. Both sides are compared at `f32`, which is what the Metal pipeline
//! computes in: past `n ≈ 4096` the `f32` eigenvalues themselves lose the
//! long-memory structure (the covariance kernel is a second difference of
//! `k^{2H}`, and in `f32` that cancellation is total from `k ≈ 4000`), so
//! agreement with the host is the invariant that survives the whole sweep
//! and agreement with theory is asserted only where `f32` still carries it.
//!
//! Reference: Davies & Harte (1987), "Tests for Hurst effect",
//! DOI: 10.1093/biomet/74.1.95.

#![cfg(feature = "metal")]

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_stochastic::device::Metal;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::traits::ProcessExt;

const HURST: f64 = 0.7;

/// The grid sizes that put the cut in a different place: `n = 1` is a
/// two-point transform with no stage inside the tile at all, `n ≤ 1024` fits
/// the tile whole, `n ≥ 2048` leaves stages between the tile and the
/// read-out, and the non-powers exercise the padding offset with each.
const GRIDS: [usize; 12] = [
  1, 2, 8, 100, 512, 1024, 2048, 3000, 4096, 8192, 12_000, 16_384,
];

/// Past this the `f32` covariance kernel has lost the long memory it is
/// meant to carry, on the host as much as on the device.
const THEORY_UP_TO: usize = 4_096;

/// The unit-lag autocovariance of fGn: `½(|k+1|^{2H} − 2|k|^{2H} + |k−1|^{2H})`.
fn rho(k: usize) -> f64 {
  if k == 0 {
    return 1.0;
  }
  let p = |x: f64| x.powf(2.0 * HURST);
  0.5 * (p(k as f64 + 1.0) - 2.0 * p(k as f64) + p(k as f64 - 1.0))
}

/// Sample variance and the first `lags` autocorrelations, pooled over paths
/// and taken around the known mean of zero. Subtracting a *sample* mean
/// instead biases the autocorrelation of a long-memory series downward by
/// several percent at these path counts — the estimator, not the sampler.
fn moments(paths: &[Vec<f64>], lags: usize) -> (f64, Vec<f64>) {
  let count = paths.iter().map(|p| p.len()).sum::<usize>() as f64;
  let var = paths.iter().flatten().map(|x| x * x).sum::<f64>() / count;
  let acf = (1..=lags)
    .map(|k| {
      let mut s = 0.0;
      let mut c = 0.0;
      for p in paths {
        for i in 0..p.len().saturating_sub(k) {
          s += p[i] * p[i + k];
          c += 1.0;
        }
      }
      if c == 0.0 { f64::NAN } else { s / c / var }
    })
    .collect();
  (var, acf)
}

fn paths_of(n: usize, on_device: bool) -> Vec<Vec<f64>> {
  let m = (600_000 / n).max(64);
  let fgn = Fgn::<f32, _>::new(HURST as f32, n, Some(1.0), Deterministic::new(7));
  let widen = |row: &[f32]| row.iter().map(|x| *x as f64).collect::<Vec<f64>>();
  if on_device {
    fgn
      .with_backend(Metal::default())
      .sample_map_view(m, |row| widen(row.as_slice().expect("contiguous row")))
  } else {
    fgn
      .sample_par(m)
      .iter()
      .map(|r| widen(r.as_slice().expect("contiguous")))
      .collect()
  }
}

#[test]
fn every_tile_cut_agrees_with_the_host_law() {
  for n in GRIDS {
    let device = paths_of(n, true);
    assert!(
      device.iter().flatten().all(|x| x.is_finite()),
      "n = {n}: the device produced a non-finite value"
    );
    if n < 8 {
      continue;
    }
    let host = paths_of(n, false);
    let (dv, da) = moments(&device, 2);
    let (hv, ha) = moments(&host, 2);
    assert!(
      (dv / hv - 1.0).abs() < 0.05,
      "n = {n}: device variance {dv:e} against the host's {hv:e}"
    );
    for (k, (d, h)) in da.iter().zip(ha.iter()).enumerate() {
      assert!(
        (d - h).abs() < 0.03,
        "n = {n}: device lag-{} autocorrelation {d:.4} against the host's {h:.4}",
        k + 1
      );
    }

    if n <= THEORY_UP_TO {
      let want = (1.0 / n as f64).powf(2.0 * HURST);
      assert!(
        (dv / want - 1.0).abs() < 0.1,
        "n = {n}: variance {dv:e} against {want:e}"
      );
      for (k, got) in da.iter().enumerate() {
        assert!(
          (got - rho(k + 1)).abs() < 0.05,
          "n = {n}: lag-{} autocorrelation {got:.4} against {:.4}",
          k + 1,
          rho(k + 1)
        );
      }
    }
  }
}

/// The map form reads the device's own output buffer rather than a copy of
/// it, and reads it in parallel, so the vector it returns has to stay indexed
/// by path and not by whichever chunk finished first. Pinned against the
/// owning batch, over a budget small enough to force several launches. Each
/// call is reseeded first: a batch draws one launch seed from the process and
/// that draw advances the stream, so two calls in a row are meant to differ.
#[test]
fn the_lending_map_returns_rows_in_path_order() {
  const M: usize = 257;
  for n in [512usize, 4096] {
    let budget = (4 * n + n) * 4 * 40;
    for device in [Metal::default(), Metal::default().with_batch_budget(budget)] {
      let fgn =
        Fgn::<f32, _>::new(HURST as f32, n, Some(1.0), Deterministic::new(13)).with_backend(device);
      fgn.seed.reseed(13);
      let mapped = fgn.sample_map_view(M, |row| (row[0], row[row.len() - 1]));
      fgn.seed.reseed(13);
      let owned = fgn.sample_par(M);
      assert_eq!(mapped.len(), M);
      for (i, (first, last)) in mapped.iter().enumerate() {
        assert_eq!(*first, owned[i][0], "n = {n}: row {i} starts elsewhere");
        assert_eq!(*last, owned[i][n - 1], "n = {n}: row {i} ends elsewhere");
      }
    }
  }
}

/// The read-out is written by the same threads that finish the transform, so
/// a grid whose last entry falls on a different thread than its first is the
/// case a fused read-out can get wrong. Every entry has to be there.
#[test]
fn the_read_out_covers_every_entry_of_the_grid() {
  for n in GRIDS {
    let fgn = Fgn::<f32, _>::new(HURST as f32, n, Some(1.0), Deterministic::new(11))
      .with_backend(Metal::default());
    let rows = fgn.sample_par(3);
    for (r, row) in rows.iter().enumerate() {
      assert_eq!(row.len(), n, "n = {n}: row {r} has the wrong length");
      assert!(
        row.iter().all(|x| x.is_finite() && *x != 0.0),
        "n = {n}: row {r} has an entry the read-out never wrote"
      );
    }
  }
}
