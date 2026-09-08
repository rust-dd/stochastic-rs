//! What the Metal fGN pipeline is held to now that it is no longer held to
//! the bytes of the pipeline it replaced.
//!
//! Three things can break silently and none of them shows up in a per-path
//! marginal. **The cut**: the low butterfly stages run inside a threadgroup
//! tile and the ones above it run two to a dispatch, and where the boundary
//! falls depends on the transform length and on how wide a threadgroup the
//! compiled kernel admits — a wrong index in either path still yields
//! plausible Gaussian numbers and destroys only the autocovariance. **The
//! pair**: one transform serves two rows, its real and imaginary halves,
//! which are two independent fGn paths only because the input noise is
//! proper complex; get a factor wrong and the two rows are correlated with
//! the right marginal law, which every per-path check passes. **The
//! chunking**: rows pair up, so a batch cut at an odd row has to compute the
//! transform in front of it and drop the half it does not own.
//!
//! Everything here is checked in `f32`, which is what the pipeline computes
//! in, and against the closed form rather than against a previous
//! implementation.
//!
//! Reference: Davies & Harte (1987), "Tests for Hurst effect",
//! DOI: 10.1093/biomet/74.1.95; Dietrich & Newsam (1997), SIAM J. Sci.
//! Comput. 18(4), DOI: 10.1137/S1064827592240555.

#![cfg(feature = "metal")]

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_stochastic::device::Metal;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::traits::ProcessExt;

const HURST: f64 = 0.7;

/// The grid sizes that put the cut in a different place: `n = 1` is a
/// two-point transform with no stage inside the tile at all, `n ≤ 1024` fits
/// the tile whole, `n ≥ 2048` leaves a radix-4 pair above it, and the
/// non-powers exercise the padding offset with each.
const GRIDS: [usize; 12] = [
  1, 2, 8, 100, 512, 1024, 2048, 3000, 4096, 8192, 12_000, 16_384,
];

/// The autocorrelation of fGn at lag `k`:
/// `½(|k+1|^{2H} − 2|k|^{2H} + |k−1|^{2H})`.
fn rho(k: usize) -> f64 {
  if k == 0 {
    return 1.0;
  }
  let p = |x: f64| x.powf(2.0 * HURST);
  0.5 * (p(k as f64 + 1.0) - 2.0 * p(k as f64) + p(k as f64 - 1.0))
}

/// Second moment and the first `lags` autocorrelations, pooled over `rows`
/// and taken around the known mean of zero. Subtracting a *sample* mean
/// instead biases the autocorrelation of a long-memory series downward by
/// several percent at these row counts — the estimator, not the sampler.
fn moments(rows: &[Vec<f64>], lags: usize) -> (f64, Vec<f64>) {
  let count = rows.iter().map(|r| r.len()).sum::<usize>() as f64;
  let var = rows.iter().flatten().map(|x| x * x).sum::<f64>() / count;
  let acf = (1..=lags)
    .map(|k| {
      let mut s = 0.0;
      let mut c = 0.0;
      for r in rows {
        for i in 0..r.len().saturating_sub(k) {
          s += r[i] * r[i + k];
          c += 1.0;
        }
      }
      if c == 0.0 { f64::NAN } else { s / c / var }
    })
    .collect();
  (var, acf)
}

fn device(n: usize, seed: u64) -> Fgn<f32, Deterministic, Metal> {
  Fgn::<f32, _>::new(HURST as f32, n, Some(1.0), Deterministic::new(seed))
    .with_backend(Metal::default())
}

fn widen(rows: Vec<ndarray::Array1<f32>>) -> Vec<Vec<f64>> {
  rows
    .iter()
    .map(|r| r.iter().map(|x| *x as f64).collect())
    .collect()
}

#[test]
fn every_tile_cut_reproduces_the_fgn_covariance() {
  for n in GRIDS {
    let m = (600_000 / n).max(64);
    let rows = widen(device(n, 7).sample_par(m));
    assert!(
      rows.iter().flatten().all(|x| x.is_finite()),
      "n = {n}: the device produced a non-finite value"
    );
    if n < 16 {
      continue;
    }

    let want_var = (1.0 / n as f64).powf(2.0 * HURST);
    let (var, acf) = moments(&rows, 4);
    assert!(
      (var / want_var - 1.0).abs() < 0.08,
      "n = {n}: variance {var:e} against {want_var:e}, off by {:.1}%",
      100.0 * (var / want_var - 1.0).abs()
    );
    for (k, got) in acf.iter().enumerate() {
      let lag = k + 1;
      assert!(
        (got - rho(lag)).abs() < 0.03,
        "n = {n}: lag-{lag} autocorrelation {got:.4} against {:.4}",
        rho(lag)
      );
    }

    // Each half of the pair on its own: a bug in one of them averages away
    // in the pooled statistic above.
    for (label, half) in [("real", 0), ("imaginary", 1)] {
      let side = rows
        .iter()
        .skip(half)
        .step_by(2)
        .cloned()
        .collect::<Vec<_>>();
      let (v, a) = moments(&side, 1);
      assert!(
        (v / want_var - 1.0).abs() < 0.1,
        "n = {n}: the {label} half's variance {v:e} against {want_var:e}"
      );
      assert!(
        (a[0] - rho(1)).abs() < 0.04,
        "n = {n}: the {label} half's lag-1 autocorrelation {:.4} against {:.4}",
        a[0],
        rho(1)
      );
    }
  }
}

/// Pooled cross-correlation between the two rows a transform serves.
fn cross(pairs: &[(Vec<f64>, Vec<f64>)], lag: isize) -> f64 {
  let mut s = 0.0;
  let mut c = 0.0;
  let mut vx = 0.0;
  let mut vy = 0.0;
  for (x, y) in pairs {
    for i in 0..x.len() {
      vx += x[i] * x[i];
      vy += y[i] * y[i];
      let j = i as isize + lag;
      if j >= 0 && (j as usize) < y.len() {
        s += x[i] * y[j as usize];
        c += 1.0;
      }
    }
  }
  let n = pairs.iter().map(|p| p.0.len()).sum::<usize>() as f64;
  s / c / ((vx / n) * (vy / n)).sqrt()
}

/// The device's own keyed draw, recomputed on the host.
///
/// `u01x4` mixes the cell number under the launch seed and reads two 64-bit
/// results as four uniforms; the pipeline turns them into one complex normal
/// through two Box-Muller transforms. Mirrored here in `f64` so the reference
/// below is an independent computation of the same definition rather than a
/// second copy of the same arithmetic.
fn u01x4(cell: u64, seed: u32) -> [f64; 4] {
  fn sm64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
  }
  fn u01(x: u32) -> f64 {
    (((x >> 8) as f32 + 0.5) / 16_777_216.0) as f64
  }
  let k = cell ^ (seed as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
  let h1 = sm64(k);
  let h2 = sm64(k ^ 0xd1b5_4a32_d192_ed03);
  [
    u01(h1 as u32),
    u01((h1 >> 32) as u32),
    u01(h2 as u32),
    u01((h2 >> 32) as u32),
  ]
}

/// Entry `j` of the transform absolute index `u` computes, from the
/// definition: the circulant embedding's eigenvalues times a proper complex
/// normal, summed directly against `exp(-2πijk/M)`. A direct sum rather than
/// a transform of its own, so nothing about the device's factorisation — its
/// tile, its radix-4 pairs, its bit-reversal — is assumed by the thing
/// checking it. One entry at a time, because that is what makes it affordable
/// at the grid sizes where the pipeline has stages between the tile and the
/// read-out.
fn reference_entry(
  fgn: &Fgn<f32, Deterministic, Metal>,
  seed: u32,
  u: usize,
  j: usize,
) -> (f64, f64) {
  let traj = 2 * fgn.n;
  let (mut xr, mut xi) = (0.0, 0.0);
  for k in 0..traj {
    let q = u01x4(2 * u as u64 * traj as u64 + k as u64, seed);
    let eig = fgn.sqrt_eigenvalues[k] as f64;
    let zr = (-2.0 * (q[0] + 1e-10).ln()).sqrt() * (std::f64::consts::TAU * q[1]).cos() * eig;
    let zi = (-2.0 * (q[2] + 1e-10).ln()).sqrt() * (std::f64::consts::TAU * q[3]).cos() * eig;
    let a = -std::f64::consts::TAU * (j * k % traj) as f64 / traj as f64;
    let (sn, cs) = a.sin_cos();
    xr += zr * cs - zi * sn;
    xi += zr * sn + zi * cs;
  }
  let out_size = fgn.n - fgn.offset;
  let scale = (out_size.max(1) as f64).powf(-HURST);
  (xr * scale, xi * scale)
}

/// The whole pipeline against the definition it implements.
///
/// This is the test that replaces bit-identity, and the reason it has to
/// exist is worth stating: a statistic cannot tell a correct realisation from
/// a plausible wrong one. Measured on this pipeline, a transform value
/// exported from the wrong position of the same transform, and a twiddle
/// whose phase advances at the wrong rate, both leave the variance, the
/// autocorrelation at every lag and the self-similar scaling of block sums
/// inside their sampling error — the circulant transform is stationary in its
/// own index, so a great many wrong answers still have the right law. What
/// separates them is recomputing the definition and comparing entry against
/// entry.
///
/// The positions are chosen where an index error surfaces: the two ends of
/// the row, and the quarter boundaries, which is where the radix-4 read-out's
/// four exported positions meet. The device works in `f32` through the whole
/// transform, so the agreement asked for is relative to the batch's scale.
#[test]
fn the_device_rows_are_the_transform_they_claim_to_be() {
  for (n, m) in [
    (64usize, 6usize),
    (128, 5),
    (300, 4),
    (512, 3),
    (4096, 3),
    (16_384, 2),
  ] {
    let out_size = n;
    let probes = [
      1,
      2,
      3,
      out_size / 4,
      out_size / 2 - 1,
      out_size / 2,
      out_size / 2 + 1,
      3 * out_size / 4,
      out_size - 1,
      out_size,
    ];
    for budget_rows in [0usize, 1, 3] {
      let handle = if budget_rows == 0 {
        Metal::default()
      } else {
        Metal::default().with_batch_budget((2 * n.next_power_of_two() + n) * 4 * budget_rows)
      };
      let seed = Deterministic::new(31).seed_value() as u32;
      let fgn =
        Fgn::<f32, _>::new(HURST as f32, n, Some(1.0), Deterministic::new(31)).with_backend(handle);
      let rows = fgn.sample_par(m);
      let rms = (rows
        .iter()
        .flatten()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        / (m * n) as f64)
        .sqrt();

      for u in 0..m.div_ceil(2) {
        for j in probes {
          let (re, im) = reference_entry(&fgn, seed, u, j);
          for (half, want) in [(0usize, re), (1, im)] {
            let row = 2 * u + half;
            if row >= m {
              continue;
            }
            let got = rows[row][j - 1] as f64;
            assert!(
              (got - want).abs() < 3e-4 * rms,
              "n = {n}, {} rows a chunk: row {row} entry {} is {got:.6e}, the transform says {want:.6e} (rms {rms:.2e})",
              if budget_rows == 0 { m } else { budget_rows },
              j - 1
            );
          }
        }
      }
    }
  }
}

/// Cross-correlation between column `i` of the first row of each pair and
/// column `i` of the second, pooled over pairs only.
fn column_corr(cols: &[[f32; 2]]) -> f64 {
  let mut sxy = 0.0;
  let mut sxx = 0.0;
  let mut syy = 0.0;
  for c in cols {
    let (x, y) = (c[0] as f64, c[1] as f64);
    sxy += x * y;
    sxx += x * x;
    syy += y * y;
  }
  sxy / (sxx * syy).sqrt()
}

/// The real and imaginary halves of a circulant-embedding transform are two
/// independent fGn paths — their cross-covariance is `Σ λ_k sin(2π(j−l)k/M)`,
/// which vanishes term by term because the eigenvalues of a symmetric
/// circulant are symmetric (Dietrich & Newsam 1997). That holds only because
/// the input noise is *proper* complex, and it is the one thing about pairing
/// no per-path statistic can see.
///
/// **It also cannot be seen by pooling a cross-correlation over lags**, which
/// is worth spelling out because it is the obvious test and it does not work.
/// Feed the two halves the same normal — `Z_k = √λ_k a_k (1+i)` — and the
/// output is `Re = ReW − ImW`, `Im = ReW + ImW`: each half still has variance
/// `γ_0` and autocovariance `γ_{j−l}` exactly, so every marginal check passes,
/// and their cross-covariance is `γ_{j+l}`, which depends on the *sum* of the
/// indices. Averaged over a fixed lag that is `O(n^{2H−2})` — 0.013 at
/// `n = 512`, under any band a pooled test could safely use. Measured: the
/// pooled version of this test passes on that mutation.
///
/// What separates them is the same-index cross-correlation taken column by
/// column, which is `ρ(2i)` under the mutation and zero under the real thing:
/// **1.0 at the first column**. Twenty thousand pairs put its standard error
/// at 1/√20000 ≈ 0.007, so the band below is five of them and the mutation
/// misses it by a factor of thirty. The lag sweep stays as a second net for
/// the failures that *are* lag-shaped — the second row being the first one
/// again, or read backwards.
#[test]
fn the_two_rows_of_a_pair_are_independent() {
  const N: usize = 256;
  const PAIRS: usize = 20_000;
  let probes = [0usize, 1, 2, 4, 8, N - 1];
  let rows = device(N, 19).sample_map_view(2 * PAIRS, |r| probes.map(|i| r[i]));
  for (k, i) in probes.iter().enumerate() {
    let cols = rows
      .chunks_exact(2)
      .map(|c| [c[0][k], c[1][k]])
      .collect::<Vec<_>>();
    let c = column_corr(&cols);
    assert!(
      c.abs() < 0.035,
      "the halves of a pair correlate {c:.4} at column {i}, expected ≈ 0"
    );
  }

  const M: usize = 2_000;
  let full = widen(device(N, 21).sample_par(2 * M));
  let pairs = full
    .chunks_exact(2)
    .map(|c| (c[0].clone(), c[1].clone()))
    .collect::<Vec<_>>();
  for lag in [-2isize, -1, 0, 1, 2] {
    let c = cross(&pairs, lag);
    assert!(
      c.abs() < 0.02,
      "the halves of a pair correlate {c:.4} at lag {lag}, expected ≈ 0"
    );
  }

  let reversed = pairs
    .iter()
    .map(|(x, y)| (x.clone(), y.iter().rev().copied().collect::<Vec<_>>()))
    .collect::<Vec<_>>();
  let c = cross(&reversed, 0);
  assert!(
    c.abs() < 0.02,
    "a pair's second half correlates {c:.4} with the first read backwards"
  );
  for (x, y) in &pairs {
    assert_ne!(x, y, "a pair's two halves are the same path");
  }

  // Neighbouring pairs come from different transforms and must be just as
  // uncorrelated, which is what says the cell numbering advanced.
  let across = full
    .windows(2)
    .skip(1)
    .step_by(2)
    .map(|w| (w[0].clone(), w[1].clone()))
    .collect::<Vec<_>>();
  let c = cross(&across, 0);
  assert!(
    c.abs() < 0.02,
    "rows of adjacent transforms correlate {c:.4}"
  );
}

/// Rows pair up, so a batch whose length or whose chunk boundary is odd is
/// the case the pairing can get wrong: the launch has to compute the
/// transform in front of it, drop the half it does not own, and still return
/// exactly the rows asked for.
#[test]
fn odd_batches_and_odd_chunks_equal_one_launch() {
  for n in [512usize, 4096] {
    for m in [1usize, 2, 3, 7, 9, 101] {
      let whole = device(n, 23).sample_par(m);
      assert_eq!(whole.len(), m, "n = {n}, m = {m}: wrong path count");
      for (i, row) in whole.iter().enumerate() {
        assert_eq!(
          row.len(),
          n,
          "n = {n}, m = {m}: row {i} has the wrong length"
        );
        assert!(
          row.iter().all(|x| x.is_finite() && *x != 0.0),
          "n = {n}, m = {m}: row {i} has an entry the read-out never wrote"
        );
      }
      for i in 0..whole.len() {
        for j in i + 1..whole.len() {
          assert_ne!(
            whole[i], whole[j],
            "n = {n}, m = {m}: rows {i} and {j} agree"
          );
        }
      }

      // Budgets that cut the batch after an odd number of rows, so a chunk
      // starts on the imaginary half of a transform.
      for per_chunk in [1usize, 3, 5] {
        let budget = (2 * n + n) * 4 * per_chunk;
        let chunked = Fgn::<f32, _>::new(HURST as f32, n, Some(1.0), Deterministic::new(23))
          .with_backend(Metal::default().with_batch_budget(budget))
          .sample_par(m);
        assert_eq!(
          whole, chunked,
          "n = {n}, m = {m}: {per_chunk} rows a chunk moved the batch"
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
    let budget = (2 * n + n) * 4 * 41;
    for handle in [Metal::default(), Metal::default().with_batch_budget(budget)] {
      let fgn =
        Fgn::<f32, _>::new(HURST as f32, n, Some(1.0), Deterministic::new(13)).with_backend(handle);
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
