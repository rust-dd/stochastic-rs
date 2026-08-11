# Migration Guide

Breaking changes are recorded here as they land, grouped by release. Entries
under `## Unreleased` describe changes on `main` that have not shipped yet.

## Unreleased

### stochastic-rs-distributions: one seeded stream, honest signatures

- `fill_slice(rng, out)` → `fill_slice(out)`. Every `Simd*` distribution's
  bulk-fill method dropped its `Rng` argument. The argument was ignored for
  24 of 27 types already (they always drew from the internal SIMD stream
  seeded at construction); `SimdBinomial`, `SimdHypergeometric`, and
  `SimdPoisson` used to honor it — meaning `sample_n`, which handed those
  three types a fresh globally-seeded `SimdRng`, silently ignored the
  `Deterministic` seed passed to their constructors. All three now draw
  from their own internal stream like every other type. Callers passing an
  explicit `Rng` (e.g. `dist.fill_slice(&mut rng, &mut out)`) drop that
  first argument; there is no other change needed since the internal
  stream was always the one actually driving output for every other type.
- `rand_distr::Distribution::sample(&self, rng)` is unchanged in shape but
  the `rng` argument is now uniformly unused across all 27 `Simd*` types
  (previously true for 24/27; `SimdBinomial`, `SimdHypergeometric`, and
  `SimdPoisson` now match). Each impl documents this; construct with
  `Deterministic::new(seed)` for reproducible output regardless of what
  `Rng` is passed to `.sample()`.
- `DistributionSampler::sample_matrix`'s parallel fan-out is now
  reproducible under `Deterministic` seeding, including across repeated
  calls on the same object. Previously each rayon worker received a
  `Clone` of the sampler, and every `Simd*` `Clone` impl re-seeds from
  `Unseeded` by design (`Clone` means "give me an independent stream") —
  so a `Deterministic`-seeded sampler silently lost reproducibility the
  moment `sample_matrix` went multi-threaded. Workers now come from a new
  `#[doc(hidden)] DistributionSampler::fork(stream_idx)` that derives each
  worker's seed from a basis value drawn fresh off the sampler's own live
  state — an interior-mutable cell distinct from the stream driving real
  samples — on every call that takes the parallel path, combined with
  `stream_idx` via `splitmix64(basis ^ stream_idx)`. Two
  identically-`Deterministic`-seeded samplers now produce bit-identical
  `sample_matrix` output call-for-call (first call matches first, second
  matches second, ...) regardless of thread count; repeated calls on the
  *same* sampler never replay, for `Deterministic`- and
  `Unseeded`-constructed samplers alike; and a serial call (below the
  parallel threshold) never touches the fork basis, so interleaving
  serial and parallel calls stays deterministic across two
  identically-seeded samplers. No API signature changed; this is a
  behavior fix.
- The Python bindings' `sample_par(m, n)` inherits this fix directly:
  seeded (`seed=...`) callers previously always executed the serial path
  under the hood (a workaround for the same-call-replay behavior above —
  going parallel for a reproducible sampler wasn't safe yet); they now
  take the same parallel path as unseeded callers, reproducible
  call-for-call via the per-call fork basis described above.
- Integer-count distributions (`SimdBinomial`, `SimdGeometric`,
  `SimdHypergeometric`, `SimdPoisson`) no longer silently emit `0` when a
  draw overflows the requested output integer type (e.g. sampling
  `Binomial(n=300, ..)` into a `u8` buffer). Overflowing draws now saturate
  to the type's `MAX` and trip a `debug_assert!` in debug builds. Code that
  relied on silent-zero overflow (there should be none — it was a
  correctness bug) will now see saturated values instead; size output
  buffers to fit the distribution's support.
- `stochastic_rs_core::simd_rng::SeedExt` gained a new required method,
  `seed_value(&self) -> u64`. This only affects code implementing
  `SeedExt` directly (no in-tree implementors besides `Unseeded` and
  `Deterministic`); consumers of `Unseeded`/`Deterministic` are
  unaffected.
- `SimdGed` and `SimdGev` now implement `DistributionSampler<T>` (were
  previously missing from the trait's coverage despite having the same
  internal-stream shape as every other float distribution) — additive,
  not breaking.
