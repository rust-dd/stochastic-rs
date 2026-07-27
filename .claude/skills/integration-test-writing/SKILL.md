---
name: integration-test-writing
description: Conventions for writing integration tests in stochastic-rs. Pinned-seed mandate, golden-numerics paper-reference pattern, no-debug-test rule, feature-gating discipline. Invoke when adding tests under tests/ or when a test is flaky.
---

# Integration test writing — stochastic-rs

This SKILL codifies the testing conventions that prevent the common
failure modes the audits caught:

- **§4.6 trap (rc.0):** a test compiled fine on `--features openblas`
  but failed on `--no-default-features` because the test imported a
  helper from an openblas-gated module without gating itself.
- **Fukasawa flake (rc.1):** a non-seeded `Fou::new` inside a test
  produced sporadic CI failures; rc.2 fixed by switching to
  `Fou::seeded(...)`.
- **§6.1 drift:** a benchmark dragged a "test helper" into the bench
  binary, kept compiling for 6 months without anyone running it, then
  broke when the underlying API changed.

The rules here keep the suite reproducible, gateable, and pruned.

## 1. The pinned-seed mandate

**Every test that draws random numbers must seed its RNG.** No
`thread_rng()`, no `rand::rng()`, no `OsRng::default()`, no
`random::<f64>()`.

### 1.1 The seed goes through the constructor, never through an `Rng` argument

This is the trap that cost a red CI on main. The workspace's SIMD
distributions take an `Rng` parameter on `fill_slice` / `sample` **and
ignore it** — they draw from their own internal stream, seeded at
construction:

```rust
pub fn fill_slice<Rr: Rng + ?Sized>(&self, _rng: &mut Rr, out: &mut [T]) {
    self.fill_slice_fast(out)   // note the underscore on _rng
}
```

So this produces a *different sample on every run* despite looking seeded:

```rust
// WRONG — the StdRng is discarded; `&Unseeded` reseeds from entropy each run
let dist = SimdNormal::<f64>::new(0.0, 1.0, &Unseeded);
let mut rng = StdRng::seed_from_u64(42);
dist.fill_slice(&mut rng, &mut x);
```

```rust
// RIGHT — the seed reaches the stream that is actually used
let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(42));
dist.fill_slice_fast(&mut x);
```

`anderson_darling.rs` shipped the wrong form for months with a comment
claiming it defended against flakiness. It did not; it was three unseeded
draws. If a test hands an `Rng` to a `Simd*` distribution, that is a bug.

### 1.2 Statistical assertions need more than one seed

A correct hypothesis test still rejects a true null at rate `alpha`, and
the SIMD stream differs between platforms — so a seed verified on
aarch64 is an unverified coin flip on the x86_64 CI runner. For any
assertion of the form "p-value must be large" or "estimate must be close",
run **three pinned seeds and assert on the best**:

```rust
let best_p = [2718u64, 999, 42]
  .into_iter()
  .map(|seed| run_test(normal_sample(seed, 700)).p_value)
  .fold(0.0_f64, f64::max);
assert!(best_p > 0.01, "every seed gave p <= 0.01 (best {best_p}); likely a bug");
```

That puts false failures near 1e-6 on any target while staying bit-exact
per platform.

### 1.3 Prefer a robust estimator over a lenient threshold

When a rate or slope is estimated from noisy level data, fix the
*estimator*, not the tolerance. `mlmc_convergence_rates_and_bs_comparison`
averaged consecutive log-ratios, which amplifies noise wherever the
denominator is small; across six seeds its α spanned 0.60 to 2.02. A
least-squares slope over all levels (Giles 2015 §2.1) held 1.02 to 1.58
on the same data.

For processes that take a seed source as the last constructor argument:

```rust
let fou = Fou::new(0.3, 0.001, -3.2, 1.0, 500, None, None, Deterministic::new(42));
```

Pin the seed at a value where the test passes with margin > 5σ — the
roll-spread test in `microstructure/spread.rs` documents seed = 11
because that's where the empirical mean lands well within tolerance.

## 2. Golden numerics from a paper

When a test asserts agreement with a paper's published table:

```rust
/// Reference: Heston (1993), Table 1, row 4 — call price 6.8061 at
/// (S=100, K=100, T=0.5, σ²=0.04, κ=2, θ=0.04, σ_v=0.5, ρ=-0.5).
#[test]
fn heston_table1_row4() {
    let pricer = HestonPricer::new(100.0, 0.0, 100.0, 0.04, 2.0, 0.04, 0.5, -0.5, 0.5);
    let call = pricer.calculate_price();
    assert!(
        (call - 6.8061).abs() < 5e-3,
        "call = {call}, expected 6.8061 ± 5e-3"
    );
}
```

Cite the paper + table + row in the doc comment so a future reader can
verify the constant against the published source. Mark the tolerance
explicitly (`5e-3` here corresponds to fewer than 0.1 % relative
error).

## 3. No `#[test] fn debug_*`

Do not commit ad-hoc `fn debug_thing` test functions for one-off
investigations. They:
- Run in CI and slow the suite.
- Drift from current API and break months later.
- Litter the test output.

If you wrote one for debugging, delete it before committing. Real
regression tests stay; debug breadcrumbs go.

## 4. Feature-gating discipline

If a test depends on `feature = "openblas"` (e.g. uses
`ndarray-linalg::SVD`), gate the *test* explicitly:

```rust
#[cfg(feature = "openblas")]
#[test]
fn svd_based_test() { /* ... */ }
```

If the entire test module depends on a feature, gate the module:

```rust
#![cfg(feature = "openblas")]
mod openblas_tests {
    // ...
}
```

The §4.6 trap was a test that imported `crate::openblas::helper` —
which only existed under `--features openblas` — without an explicit
gate. The test compiled fine because the surrounding suite implicitly
had openblas enabled, then broke when someone ran with
`--no-default-features`.

Verification: the release-checklist mandates `cargo test --workspace
--no-default-features` to catch missing gates.

## 5. Tests in `tests/` vs in-module `#[cfg(test)]`

| Location | Use for |
|---|---|
| `src/foo.rs` `#[cfg(test)] mod tests { ... }` | Unit tests of the module's own functions |
| `tests/foo_integration.rs` | Cross-crate / cross-module integration |

Workspace integration tests live in `tests/` directories at each
sub-crate root and in the umbrella crate. They use only the public
API (no `crate::` access) and exercise multi-module flows like
"calibrate a Heston, build a vol surface, price a basket".

## 6. Tolerance taxonomy

| Domain | Typical tolerance |
|---|---|
| Closed-form analytic (matches paper) | 1e-12 (float64 noise) |
| Iterative analytic (Brent root, Padé exp) | 1e-9 |
| Monte Carlo with N=10⁵ paths | 1e-2 (1 % of price) |
| Monte Carlo with N=10⁴ paths | 1e-1 (10 %) |
| Calibrator RMSE on synthetic data | 1e-3 |
| Pinned-seed estimator on n=10⁴ samples | depends on σ/√N — compute |

Don't pad the tolerance to make the test pass on a flaky path.
Either:
- Increase N until the test passes within the natural tolerance.
- Fix the test to compare a less-noisy summary statistic.
- Accept the test is fundamentally noisy and document the seed
  selection rationale.

## 7. Things that should always panic (and prove it)

For invariants that must hold:

```rust
#[test]
#[should_panic(expected = "n must be at least 2")]
fn rejects_n_below_two() {
    let _ = Fou::<f64>::new(0.5, 0.0, 1.0, 0.1, 1, None, None);
}
```

The `expected = "..."` string anchor is **mandatory**. Without it,
`#[should_panic]` accepts any panic, including the wrong one (e.g. a
later index-out-of-bounds inside the constructor masking the missing
validation).

## 8. Common patterns

### Comparing to scipy / R

If a test compares to a Python / R reference, embed the exact command
in the doc comment:

```rust
/// Reference: scipy.linalg.expm of [[-0.1, 0.1], [0.05, -0.05]],
/// computed via:
///   import scipy.linalg; import numpy as np
///   scipy.linalg.expm(np.array([[-0.1, 0.1], [0.05, -0.05]]))
/// Result: [[0.90713, 0.09287], [0.04643, 0.95357]]
#[test]
fn expm_matches_scipy() { /* ... */ }
```

Reproducibility from the doc comment alone is the standard.

### Round-trip tests

For serialisation, parameter parsing, type conversions:

```rust
#[test]
fn params_roundtrip_through_dvector() {
    let p1 = SabrParams::new(0.2, 0.5, 0.3, -0.5);
    let v: DVector<f64> = p1.clone().into();
    let p2 = SabrParams::from(v);
    assert_eq!(p1, p2);
}
```

### Property-based shape tests

For variance / covariance / monotonicity properties:

```rust
#[test]
fn fbm_variance_scales_as_t_2h() {
    for h in [0.3, 0.5, 0.7] {
        for t in [0.5, 1.0, 2.0] {
            let var_empirical = mc_variance(h, t, n_paths = 50_000, seed = 42);
            let var_theory = t.powf(2.0 * h);
            assert!(
                ((var_empirical - var_theory) / var_theory).abs() < 0.05,
                "h={h}, t={t}: emp={var_empirical}, theory={var_theory}"
            );
        }
    }
}
```

## 9. Anti-patterns

- **Do not** rely on `thread_rng()` for any test that asserts a
  numeric.
- **Do not** commit `fn debug_*` tests.
- **Do not** use `#[should_panic]` without the `expected = "..."`
  anchor.
- **Do not** loosen tolerances to mask flakes. Fix the seed or fix the
  test.
- **Do not** add a test that imports a feature-gated symbol without
  gating the test itself.
- **Do not** put feature-gated `use` statements above the
  `#[cfg(test)]` boundary; the import must be inside the gate.

## 10. Reference suite

- `pricing/heston.rs::tests` — paper-table golden numerics.
- `vol_surface/svi.rs::tests` — closed-form derivatives + roundtrip.
- `microstructure/spread.rs::tests` — explicit seed pinning + tolerance
  rationale comments.
- `credit/migration.rs::tests` — scipy-cross-checked expm reference.

## Related SKILLs

- `bench-writing` — for `[[bench]] required-features` gating.
- `feature-flag-management` — gating discipline.
- `release-checklist` — runs the test matrix as a release gate.
