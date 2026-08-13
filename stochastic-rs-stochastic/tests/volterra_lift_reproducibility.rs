//! Permanent reproducibility oracle for the four `Rl*` Markov-lift
//! processes.
//!
//! Pins every value in [`RlBlackScholes::sample`]/`sample_batch`,
//! [`RlFBm::sample`]/`sample_batch`, [`RlFOU::sample`]/`sample_batch`, and
//! [`RlHeston::sample`]/`sample_batch` (spot *and* variance), built with
//! [`Deterministic::new(42)`](Deterministic::new) and the same parameters
//! `tests/reproducibility_all_processes/rough.rs` uses, against the
//! `f64::to_bits()` pattern captured on the tree *before* `MarkovLift` was
//! generalised over [`VolterraKernel`](stochastic_rs_stochastic::volterra::VolterraKernel)
//! (commit `1faaa99`) — compared within a **relative tolerance**, not bit
//! equality.
//!
//! **Why tolerance, not `to_bits()` equality.** Routing `MarkovLift`'s
//! boundary weights and per-mode history-sum weights through
//! [`VolterraKernel`]'s trait methods reassociates the underlying
//! arithmetic: `RlKernel::evaluate`/`integral_from_zero` divide directly
//! and evaluate $\Gamma(H{+}3/2)$ independently, where the original
//! hand-written `MarkovLift` multiplied by a cached reciprocal and reused
//! the identity $\Gamma(H{+}3/2)=(H{+}\tfrac12)\Gamma(H{+}\tfrac12)$; the
//! per-mode weights are normalised once per node and summed directly,
//! instead of summed unnormalised and scaled once at the end. Measured
//! against a 60-decimal-digit (`mpmath`) reference for a spread of
//! `(H, dt)` pairs and a representative history-sum snapshot, both
//! orderings land within single-digit ULPs of the reference on either side
//! of it — reassociation noise, not a defect (see
//! `task-2-report.md` in `.superpowers/sdd/2026-08-13-volterra-sde-engine/`
//! for the full numeric proof; a genuine double-normalisation bug would be
//! off by a whole factor of $\Gamma(H{+}\tfrac12)$, roughly 33% at
//! $H=0.1$, not ~1e-13). See `MIGRATION.md` for the before/after example.
//!
//! **The bound: `1e-11` relative.** The largest observed shift across every
//! value dumped by both `sample` and `sample_batch`, all four processes, is
//! ≈512 ULP (≈1.05e-13 relative, at `RlHeston::sample_batch`'s variance
//! output, where the reference value itself is ≈6.62e-5). The comparison
//! below (see `assert_close`) is `|actual - reference| <= 1e-11 * (1.0 +
//! |reference|)`, not a pure relative check, so the margin this bound
//! actually leaves is not `1e-11` divided by that `1.05e-13` relative
//! figure (which would say ~100×). At that worst point the absolute
//! difference is ≈6.94e-18 and the tolerance, dominated by the `+1` floor
//! since `|reference| << 1`, is ≈1.0000662e-11 — a realised margin of
//! ≈1.44e6×, not ~100×. The `+1` floor dominates this way for every value
//! in this file's pinned data with `|reference| < 1` (nearly all of it —
//! the few `RlBlackScholes`/`RlHeston`-spot values near $100$ are the
//! exception, where the tolerance is closer to purely relative and the
//! margin is correspondingly closer to the ~100× figure). Either way this
//! is tight enough that a real regression (a dropped or duplicated
//! normalising factor, a changed quadrature, a wrong sign) still fails
//! loudly — Task 2's review independently confirmed this by injecting
//! 1e-10, 3e-8, and 1e-6 relative perturbations into `RlKernel::evaluate`,
//! all caught — loose enough to absorb a modest amount of additional
//! cross-platform/cross-compiler variation in the transcendental
//! (`powf`/`exp`/`gamma`) evaluations feeding the reassociated paths, and
//! four orders of magnitude tighter than the `1e-9` the umbrella's own
//! `tests/sampler_v3_golden.rs` uses for the same underlying reason
//! (cross-computation float rounding on `powf`/FFT-heavy paths not
//! reproducing bit patterns across architectures).
//!
//! **Do not loosen this bound to make a failing change pass.** If this test
//! starts failing by more than a rounding-order amount, that is a real
//! regression to fix, not a tolerance to widen.
//!
//! The ten pinned `const [u64; N]` reference arrays live in the four
//! `volterra_lift_reproducibility/*.rs` companion data modules (one per
//! process: `bs`, `fbm`, `fou`, `heston`) rather than in this file, purely
//! to keep this file under this crate's line-count limit (the precedent,
//! `tests/sampler_v3_golden.rs`, is 482 lines by keeping its own golden
//! arrays short and inline; this oracle's ten arrays, at $N=24$/$96$ each to
//! match `sample`/`sample_batch`, are collectively too large for that same
//! approach here) — mirroring how `reproducibility_all_processes.rs` splits
//! its own guard logic across directory-named submodules for the same
//! reason. All four data modules compile into this same
//! `volterra_lift_reproducibility` test binary, so `cargo test` still runs
//! and reports all four `#[test]` functions below as one unit; the split is
//! data-vs-logic, not a change to what is tested.
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::rough::rl_bs::RlBlackScholes;
use stochastic_rs_stochastic::rough::rl_fbm::RlFBm;
use stochastic_rs_stochastic::rough::rl_fou::RlFOU;
use stochastic_rs_stochastic::rough::rl_heston::RlHeston;
use stochastic_rs_stochastic::traits::ProcessExt;

#[path = "volterra_lift_reproducibility/bs.rs"]
mod bs;
#[path = "volterra_lift_reproducibility/fbm.rs"]
mod fbm;
#[path = "volterra_lift_reproducibility/fou.rs"]
mod fou;
#[path = "volterra_lift_reproducibility/heston.rs"]
mod heston;

use bs::BS_SAMPLE;
use bs::BS_SAMPLE_BATCH;
use fbm::FBM_SAMPLE;
use fbm::FBM_SAMPLE_BATCH;
use fou::FOU_SAMPLE;
use fou::FOU_SAMPLE_BATCH;
use heston::HESTON_SAMPLE_BATCH_S;
use heston::HESTON_SAMPLE_BATCH_V;
use heston::HESTON_SAMPLE_S;
use heston::HESTON_SAMPLE_V;

/// Matches `tests/reproducibility_all_processes/rough.rs`'s own `N`.
const N: usize = 24;
/// Batch width for the `sample_batch` half of the oracle. Small on purpose
/// — this test is about arithmetic reproducibility, not statistics, and
/// every row already exercises the full cache-tiled `simulate_batch` path.
const M: usize = 4;
/// Relative tolerance, justified in the module doc above against the
/// measured maximum reassociation shift (≈512 ULP / ≈1.05e-13 relative).
const REL_TOL: f64 = 1e-11;

fn assert_close(label: &str, actual: &Array1<f64>, expected: &[u64]) {
  assert_eq!(
    actual.len(),
    expected.len(),
    "{label}: length mismatch (got {}, expected {})",
    actual.len(),
    expected.len()
  );
  for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
    let reference = f64::from_bits(e);
    let tol = REL_TOL * (1.0 + reference.abs());
    let diff = (a - reference).abs();
    assert!(
      diff <= tol,
      "{label}[{i}]: got {a:e} (0x{:016x}), reference {reference:e} (0x{:016x}) — \
       |diff|={diff:e} exceeds tol={tol:e} ({REL_TOL:e} relative)",
      a.to_bits(),
      e
    );
  }
}

fn assert_close_2d(label: &str, actual: &Array2<f64>, expected: &[u64]) {
  let flat = actual.iter().copied().collect::<Array1<f64>>();
  assert_close(label, &flat, expected);
}

#[test]
fn rl_black_scholes_reproducible() {
  let bs = RlBlackScholes::new(
    0.1,
    100.0,
    0.02,
    0.2,
    N,
    Some(1.0),
    None,
    Deterministic::new(42),
  );
  assert_close("RlBlackScholes::sample", &bs.sample(), &BS_SAMPLE);
  assert_close_2d(
    "RlBlackScholes::sample_batch",
    &bs.sample_batch(M),
    &BS_SAMPLE_BATCH,
  );
}

#[test]
fn rl_fbm_reproducible() {
  let fbm = RlFBm::new(0.1, N, Some(1.0), None, Deterministic::new(42));
  assert_close("RlFBm::sample", &fbm.sample(), &FBM_SAMPLE);
  assert_close_2d(
    "RlFBm::sample_batch",
    &fbm.sample_batch(M),
    &FBM_SAMPLE_BATCH,
  );
}

#[test]
fn rl_fou_reproducible() {
  let fou = RlFOU::new(
    0.1,
    1.0,
    0.0,
    0.3,
    N,
    Some(0.0),
    Some(1.0),
    None,
    Deterministic::new(42),
  );
  assert_close("RlFOU::sample", &fou.sample(), &FOU_SAMPLE);
  assert_close_2d(
    "RlFOU::sample_batch",
    &fou.sample_batch(M),
    &FOU_SAMPLE_BATCH,
  );
}

#[test]
fn rl_heston_reproducible() {
  let heston = RlHeston::new(
    0.1,
    Some(100.0),
    Some(0.04),
    1.5,
    0.04,
    0.3,
    -0.6,
    0.0,
    N,
    Some(1.0),
    None,
    Deterministic::new(42),
  );
  let [s, v] = heston.sample();
  assert_close("RlHeston::sample.s", &s, &HESTON_SAMPLE_S);
  assert_close("RlHeston::sample.v", &v, &HESTON_SAMPLE_V);

  let [sb, vb] = heston.sample_batch(M);
  assert_close_2d("RlHeston::sample_batch.s", &sb, &HESTON_SAMPLE_BATCH_S);
  assert_close_2d("RlHeston::sample_batch.v", &vb, &HESTON_SAMPLE_BATCH_V);
}
