//! Smoke test for the seed plumbing introduced when distributions collapsed
//! to a single `new(args, &seed)` constructor and `SimdRngExt` became the
//! generic backing-RNG interface. Verifies that:
//!
//! 1. Two `Unseeded` `Fbm` instances produce different paths each time
//!    (auto-seeded streams are independent).
//! 2. Two `Deterministic::new(seed)` `Fbm` instances with the same seed
//!    produce the same path (reproducibility holds end-to-end through the
//!    process → SimdRng pipeline).
//! 3. Different deterministic seeds produce different paths.

use stochastic_rs::prelude::*;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::simd_rng::SeedExt;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::stochastic::process::fbm::Fbm;

fn first_diff(a: &[f64], b: &[f64]) -> Option<usize> {
  a.iter()
    .zip(b.iter())
    .position(|(x, y)| (x - y).abs() > 1e-15)
}

fn main() {
  let n = 128;
  let h = 0.7_f64;
  let t = 1.0_f64;

  // ─── 1. Unseeded → independent streams ────────────────────────────────
  let fbm_a = Fbm::<f64, _>::new(h, n, Some(t), Unseeded);
  let fbm_b = Fbm::<f64, _>::new(h, n, Some(t), Unseeded);
  let path_a = fbm_a.sample().to_vec();
  let path_b = fbm_b.sample().to_vec();
  let diff_at = first_diff(&path_a, &path_b);
  println!(
    "Unseeded × 2  → first divergence at index {:?} (expect Some(_))",
    diff_at
  );
  assert!(
    diff_at.is_some(),
    "two Unseeded Fbm instances produced identical paths — seed propagation broken?"
  );

  // ─── 2. Deterministic(seed) reproducibility ──────────────────────────
  let seed = 0xDEAD_BEEFu64;
  let fbm_c = Fbm::<f64, _>::new(h, n, Some(t), Deterministic::new(seed));
  let fbm_d = Fbm::<f64, _>::new(h, n, Some(t), Deterministic::new(seed));
  let path_c = fbm_c.sample().to_vec();
  let path_d = fbm_d.sample().to_vec();
  let diff_cd = first_diff(&path_c, &path_d);
  println!(
    "Deterministic({seed}) × 2 → first divergence: {:?} (expect None — fully reproducible)",
    diff_cd
  );
  assert_eq!(
    diff_cd, None,
    "Deterministic seed did NOT reproduce — seed propagation broken"
  );

  // ─── 3. Different seeds → different paths ────────────────────────────
  let fbm_e = Fbm::<f64, _>::new(h, n, Some(t), Deterministic::new(seed));
  let fbm_f = Fbm::<f64, _>::new(h, n, Some(t), Deterministic::new(seed.wrapping_add(1)));
  let path_e = fbm_e.sample().to_vec();
  let path_f = fbm_f.sample().to_vec();
  let diff_ef = first_diff(&path_e, &path_f);
  println!(
    "Deterministic({seed}) vs Deterministic({}) → first divergence at index {:?} (expect Some(_))",
    seed.wrapping_add(1),
    diff_ef
  );
  assert!(
    diff_ef.is_some(),
    "different seeds produced identical paths — seed mixing broken"
  );

  // ─── 4. Same Fbm instance, repeated samples ──────────────────────────
  // `Fbm` stores `seed: S`, so `sample()` calls advance the internal seed
  // state through `S::rng()`. Two consecutive samples from the SAME
  // instance must therefore differ (otherwise every call would replay).
  let fbm_g = Fbm::<f64, _>::new(h, n, Some(t), Deterministic::new(seed));
  let path_g1 = fbm_g.sample().to_vec();
  let path_g2 = fbm_g.sample().to_vec();
  let diff_g = first_diff(&path_g1, &path_g2);
  println!(
    "Same instance, sample × 2 → first divergence at index {:?} (expect Some(_))",
    diff_g
  );
  assert!(
    diff_g.is_some(),
    "consecutive samples from the same Deterministic Fbm matched — seed not advancing"
  );

  // ─── 5. SeedExt::reseed — in-place seed swap, no realloc ─────────────
  // After `seed.reseed(s)` the same process instance must reproduce the
  // path it would have produced if constructed with `Deterministic::new(s)`
  // (bit-for-bit). Lets calibration loops sweep seeds without rebuilding
  // the process, which is the main motivation for the `reseed` API.
  let fbm_h = Fbm::<f64, _>::new(h, n, Some(t), Deterministic::new(0));
  fbm_h.seed.reseed(seed);
  let path_h = fbm_h.sample().to_vec();
  let diff_h = first_diff(&path_c, &path_h);
  println!(
    "Reseed(seed) on a different instance → divergence vs Deterministic(seed): {:?} (expect None)",
    diff_h
  );
  assert_eq!(
    diff_h, None,
    "reseed(seed) did NOT reproduce the Deterministic(seed) stream — reset path broken"
  );

  // 5b. After reseed, two replays of the same seed match.
  let fbm_i = Fbm::<f64, _>::new(h, n, Some(t), Deterministic::new(0));
  fbm_i.seed.reseed(123);
  let path_i1 = fbm_i.sample().to_vec();
  fbm_i.seed.reseed(123);
  let path_i2 = fbm_i.sample().to_vec();
  let diff_i = first_diff(&path_i1, &path_i2);
  println!(
    "Reseed(123) → sample → reseed(123) → sample: divergence {:?} (expect None)",
    diff_i
  );
  assert_eq!(
    diff_i, None,
    "reseeding to the same value did NOT replay the same stream"
  );

  // ─── 6. Unseeded::reseed is a no-op ───────────────────────────────────
  // Spec: `Unseeded.reseed(seed)` is silently a no-op (no fixed state to
  // assign to). The call must compile and have no observable effect.
  let fbm_j = Fbm::<f64, _>::new(h, n, Some(t), Unseeded);
  fbm_j.seed.reseed(seed); // no-op, should not panic
  let _ = fbm_j.sample();
  println!("Unseeded.reseed(...) → no panic, sample still works ✓");

  println!("\n--- Fgn (direct, not via Fbm cumsum) ---");

  // ─── 7. Fgn Unseeded → independent streams ────────────────────────────
  let fgn_a = Fgn::<f64, _>::new(h, n, Some(t), Unseeded);
  let fgn_b = Fgn::<f64, _>::new(h, n, Some(t), Unseeded);
  let p_a = fgn_a.sample().to_vec();
  let p_b = fgn_b.sample().to_vec();
  let d_ab = first_diff(&p_a, &p_b);
  println!(
    "Fgn Unseeded × 2 → first divergence at index {:?} (expect Some(_))",
    d_ab
  );
  assert!(
    d_ab.is_some(),
    "two Unseeded Fgn instances produced identical paths"
  );

  // ─── 8. Fgn Deterministic reproducibility ─────────────────────────────
  let fgn_c = Fgn::<f64, _>::new(h, n, Some(t), Deterministic::new(seed));
  let fgn_d = Fgn::<f64, _>::new(h, n, Some(t), Deterministic::new(seed));
  let p_c = fgn_c.sample().to_vec();
  let p_d = fgn_d.sample().to_vec();
  assert_eq!(
    first_diff(&p_c, &p_d),
    None,
    "Fgn Deterministic seed did not reproduce"
  );
  println!("Fgn Deterministic({seed}) × 2 → identical paths ✓");

  // ─── 9. Fgn different seeds → different paths ─────────────────────────
  let fgn_e = Fgn::<f64, _>::new(h, n, Some(t), Deterministic::new(seed));
  let fgn_f = Fgn::<f64, _>::new(h, n, Some(t), Deterministic::new(seed.wrapping_add(1)));
  let p_e = fgn_e.sample().to_vec();
  let p_f = fgn_f.sample().to_vec();
  assert!(
    first_diff(&p_e, &p_f).is_some(),
    "Fgn with different seeds produced identical paths"
  );
  println!("Fgn Deterministic(seed) vs Deterministic(seed+1) → different paths ✓");

  // ─── 10. Fgn same instance, repeated samples ──────────────────────────
  let fgn_g = Fgn::<f64, _>::new(h, n, Some(t), Deterministic::new(seed));
  let p_g1 = fgn_g.sample().to_vec();
  let p_g2 = fgn_g.sample().to_vec();
  assert!(
    first_diff(&p_g1, &p_g2).is_some(),
    "Fgn consecutive samples matched — seed not advancing"
  );
  println!("Fgn same instance × 2 samples → different paths ✓");

  // ─── 11. Fgn reseed reproduces Deterministic stream ───────────────────
  let fgn_h = Fgn::<f64, _>::new(h, n, Some(t), Deterministic::new(0));
  fgn_h.seed.reseed(seed);
  let p_h = fgn_h.sample().to_vec();
  assert_eq!(
    first_diff(&p_c, &p_h),
    None,
    "Fgn reseed(seed) did not reproduce Deterministic(seed)"
  );
  println!("Fgn reseed(seed) → identical to Deterministic(seed) stream ✓");

  // ─── 12. Fgn Unseeded::reseed is a no-op ──────────────────────────────
  let fgn_i = Fgn::<f64, _>::new(h, n, Some(t), Unseeded);
  fgn_i.seed.reseed(seed);
  let _ = fgn_i.sample();
  println!("Fgn Unseeded.reseed(...) → no panic, sample still works ✓");

  println!("\n✓ all 12 Fbm + Fgn seed propagation + reseed invariants hold");
}
