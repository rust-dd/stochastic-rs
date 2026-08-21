//! `ProcessExt` and dimensional output markers.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::parallel::prelude::*;
use stochastic_rs_distributions::traits::FloatExt;

use super::sampler::PathSampler;

/// Upper bound on the number of chunks [`ProcessExt::sample_par`] /
/// [`ProcessExt::sample_map`] split `m` paths into.
///
/// `chunk_count`'s prologue (building every chunk's sampler, sequentially,
/// on the calling thread — see [`ProcessExt::chunked_samplers`]) costs at
/// most `MAX_CHUNKS` sampler constructions, however large `m` is. That
/// bound matters because `sampler()` is not always cheap: the worst in-tree
/// case (`Cir2F`, whose `sampler()` evaluates a user-supplied `Fn1D` — a
/// Python callback, for the `Fn1D::Py` variant — once per grid point) would
/// otherwise pay a *sequential, GIL-round-tripping* construction cost that
/// grows with `m` (the previous rule, `m.div_ceil(8)`, made `sample_par(1000)`
/// build 125 chunks — 125 sequential constructions — instead of a number
/// bounded by the machine's core count). 64 is comfortably above real-world
/// hardware-thread counts, so `m >= 64` still saturates any realistic rayon
/// pool, while `m < 64` gets one chunk per path (full parallelism for small
/// `m`, the other end `m.div_ceil(8)` got wrong by forcing `m <= 8` fully
/// serial).
const MAX_CHUNKS: usize = 64;

/// Number of chunks to split `m` paths into.
///
/// A pure function of `m` alone. **Must never read
/// `rayon::current_num_threads()`**: the chunk count fixes how many times
/// [`sampler()`](ProcessExt::sampler) (and
/// [`advance_chunk_seed()`](ProcessExt::advance_chunk_seed)) is called
/// before any chunk starts running, which fixes how many times a
/// [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic) process's
/// shared seed state advances. If that count depended on the ambient
/// thread-pool size, the same seed and the same `m` could produce different
/// output on two machines (or two test runs) with different pool sizes —
/// exactly the defect this module fixes.
///
/// `pub(crate)` rather than private: the `accelerate` feature's `Backend`
/// impl (`device.rs`) reuses it verbatim for `Fgn`/`Fbm`'s own `sample_par`
/// overrides on that backend, which bypass this trait to reach the batched
/// backend path but must split `m` the same data-derived way to get the
/// same thread-count-independence guarantee. The default `Cpu` backend
/// does not reuse it — measurement showed grouping several paths per
/// `MAX_CHUNKS`-capped chunk regressed wall time roughly 2× at `m = 1000`
/// there, because each path's own FFT call is itself a nested rayon
/// parallel region (see `device.rs`'s `Cpu::generate_batch` doc), so `Cpu`
/// instead derives one basis per **path**, uncapped.
pub(crate) fn chunk_count(m: usize) -> usize {
  m.min(MAX_CHUNKS)
}

/// Splits `m` into `chunks` contiguous run lengths, as even as possible (the
/// first `m % chunks` chunks get one extra path), yielded in chunk order.
///
/// `chunks == 0` only ever arises from `chunk_count(0)`; the `checked_div`/
/// `checked_rem` fall back to `0` there (rather than relying on every caller
/// to check `m` first) so this function stays total instead of panicking on
/// the `m / 0` that a plain division would perform. `pub(crate)` for the
/// same reason as `chunk_count` above.
pub(crate) fn chunk_lens(m: usize, chunks: usize) -> impl Iterator<Item = usize> {
  let base = m.checked_div(chunks).unwrap_or(0);
  let rem = m.checked_rem(chunks).unwrap_or(0);
  (0..chunks).map(move |i| base + usize::from(i < rem))
}

/// Stochastic process simulation trait.
///
/// Each process exposes `sample()` returning a [`Self::Output`] and
/// `sample_par(m)` returning `m` independent samples via Rayon.
///
/// ## Time-horizon (`t`) convention
///
/// Most process structs hold `t: Option<T>`. When `t.is_none()`, implementations
/// fall back to `T::one()` (i.e. one unit of time, conventionally one year for
/// finance models). This matches the convention used across the workspace and
/// the audit document; do **not** rely on it implicitly for interest-rate or
/// volatility models where the horizon meaningfully drives parameter scaling
/// (Vasicek, CIR, HJM, Heston, Bergomi). For those, set `t` explicitly. Note
/// that [`crate::interest::bgm::Bgm`] despite its name is **not** a coupled
/// LMM/BGM (see its module doc); it is a parallel array of independent
/// Euler-stepped multiplicative martingales.
///
/// ## Backend selection
///
/// Re-type a process to a compile-time sampling backend with the turbofish
/// `process.on::<B>()` where `B: `[`Backend`](crate::device::Backend) (e.g.
/// `process.on::<CudaNative>()`); the backend marker propagates to the
/// process's noise source with no runtime branch. Only the fractional family
/// (built on [`Fgn`](crate::noise::fgn::Fgn)) exposes GPU backends today, and a
/// GPU marker only exists when its feature is compiled.
///
/// ## Sampling architecture
///
/// The public surface is [`sample`](Self::sample), [`sample_par`](Self::sample_par)
/// and [`sample_map`](Self::sample_map). Under them sits a hidden
/// [`PathSampler`] holding all per-call mutable state (RNG, distribution
/// buffers, scratch, precomputed scales); [`sampler`](Self::sampler) builds
/// one. The parallel methods split `m` paths into a fixed number of chunks —
/// a pure function of `m`, never of the ambient thread pool — and construct
/// **one sampler per chunk**, all sequentially on the calling thread before
/// any chunk reaches rayon. [`sample_map`](Self::sample_map) folds over each
/// chunk's paths reusing a single output buffer;
/// [`sample_par`](Self::sample_par) keeps every path, allocating each fresh
/// (no buffer reuse, no clone).
///
/// ### Reproducibility requirement on implementors
///
/// Sequential chunk construction only produces bit-identical, thread-count-
/// independent, **chunk-independent** output if `sampler()` captures its
/// basis via `self.seed.derive()` — never `self.seed.clone()`, and never by
/// reading `&self.seed` lazily per path from inside the returned sampler.
/// `derive()` advances `self.seed`'s shared state *and* hash-mixes the
/// result before handing it to the new owner, so — since
/// [`chunked_samplers`](Self::chunked_samplers) calls `sampler()` once per
/// chunk, sequentially, before any chunk reaches rayon — every chunk's
/// basis is a distinct, mutually-uncorrelated hash output. `clone()` is
/// `SeedExt`'s deliberate *non-advancing, non-mixing* inverse of that: it
/// copies the raw counter with zero hash hops, so even a process that
/// advances the shared state once per chunk before cloning (the
/// [`advance_chunk_seed`](Self::advance_chunk_seed) mechanism below) only
/// ever gives adjacent chunks bases that are one raw arithmetic stride
/// apart — close enough, for whatever the sampler's own per-path code then
/// does with that raw value, to still overlap across chunks (measured: only
/// 78 of 1000 `Sabr` paths, 67 of 256 `Heston` paths, actually distinct).
/// Reading `&self.seed` lazily per path is unsound for a third, independent
/// reason — every chunk's sampler then shares live access to the same
/// atomic, racing on it during the parallel region itself, which no amount
/// of pre-parallel sequencing can fix.
///
/// A sampler may call `.derive()` again, further, on its *own* already-
/// derived basis without any of the above risk — e.g. to build several
/// independent per-path sub-streams, or because a downstream constructor
/// needs an owned `S` rather than a borrowed `&S`
/// ([`MultifactorSabr`](crate::volatility::multifactor_sabr::MultifactorSabr)
/// builds two fresh [`Gn`](crate::noise::gn::Gn) generators per path this
/// way). That basis is already a chunk-unique hash output, so any amount of
/// further ticking, by any mechanism, stays confined to that one chunk's
/// own uncorrelated sequence.
///
/// [`advance_chunk_seed`](Self::advance_chunk_seed) exists for one
/// remaining legitimate case: a `sampler()` that clones because the clone
/// feeds a *persistent* engine (e.g. a buffered `SimdNormal`) built once per
/// chunk and reused across every path in that chunk via the engine's own
/// internal advancement, never re-consulting the `Deterministic`-level seed
/// per path — see [`CirPlusPlus`](crate::interest::cir_pp::CirPlusPlus).
/// Overriding it to advance the shared state before each chunk's
/// `sampler()` call gives each such clone a distinct starting point.
///
/// **No process in this crate is an exception, full or partial.** Every
/// concrete `ProcessExt` implementor derives all of its sampled randomness
/// from `self.seed` — its diffusion component, and, for the jump-diffusion
/// types that have one, its jump component too. That was not always true;
/// getting here took several corrected rounds (a `Cgns`- or `Fgn`-shaped
/// diffusion once thought structurally unfixable turned out to be a plain
/// missed wire — `Cgns::sample_impl<S2: SeedExt>(&self, seed: &S2)` was
/// generic over an *external* seed all along, so a bare `.sample()`
/// bypassing it was a bug, not a limitation; a `pub cpoisson:
/// CompoundPoisson<T, D>` field structurally pinned to `Unseeded` — present
/// on `Merton`, `Kou`, `LevyDiffusion`, `Bates1996` and `JumpFou` — turned
/// out to need a genuine breaking constructor change, made by the
/// zero-exception-reproducibility wave's Tasks 1 and 2). That history is
/// deliberately not repeated here — a corrected verdict restated as a live
/// exception list is exactly how this file drifted out of sync with the
/// code three times before.
///
/// **This guarantee is enforced, not merely asserted.**
/// `tests/reproducibility_all_processes.rs` enumerates every concrete
/// `ProcessExt` implementor in the crate (124 as of this wave — derived by
/// grepping `impl … ProcessExt<…> for …` blocks under `src/`, excluding the
/// blanket marker-trait impls in this file; the test's own doc comment
/// carries the exact command and an instruction to extend the list when a
/// process is added) and asserts, for each: two freshly constructed,
/// identically-[`Deterministic`](stochastic_rs_core::simd_rng::Deterministic)
/// -seeded instances agree bit-for-bit on [`sample`](Self::sample), and
/// [`sample_par`](Self::sample_par) agrees bit-for-bit across rayon
/// thread-pool sizes at both `m <= MAX_CHUNKS` (one path per chunk — cannot
/// by itself expose cross-chunk correlation) and `m > MAX_CHUNKS` (several
/// paths per chunk). Five separate instances of the missed-wire/pinned-field
/// bug class above were each found by a different ad-hoc sweep before this
/// test existed, precisely because no single test enumerated the whole
/// surface; a type with no line in that file is a type nothing here is
/// proving anything about.
///
/// **Backend-level exceptions are a separate axis, untouched by the
/// above and not covered by that test** (which only instantiates
/// backend-generic processes on the default `Cpu` backend):
/// `Fgn`/`Fbm` on the `accelerate` feature's `Accelerate` backend get
/// thread-count-independent seed *consumption* but not bit-identical
/// *output* — vDSP's own arithmetic is not bit-stable across Apple
/// Silicon's heterogeneous P-core/E-core scheduler. Measured on an M4 Max:
/// two identically-seeded `Accelerate` calls agreed in all of 400 swept
/// `(n, m)` configurations on an otherwise-idle system (0/400 differing),
/// but 21 of those same 400 configurations diverged when the identical
/// sweep ran under induced full-core load (worst observed relative
/// difference `2.08e-3`); the `Cpu` backend, swept under the identical
/// induced load, stayed bit-exact in all 400 — see
/// [`Backend`](crate::device::Backend)'s own doc for the full per-backend
/// table and `tests/deterministic_parallelism_accelerate.rs` for the
/// measurement. GPU backends (`CudaNative`/`CubeCl`/`MetalNative`) are
/// excluded from this guarantee entirely, deliberately: each draws one
/// value from `self.seed.rng()` per batch call and hands it to the
/// on-device kernel's own Philox/PCG-style RNG, so output is a function of
/// the pinned seed but *not* of host thread-pool size, yet cross-run
/// bit-identity across GPU driver versions, vendors, or repeated runs on
/// the same device is untested and not promised; `Fbm` specifically does
/// not even reach seed-dependence on those three (see
/// [`Fbm::sample_par`](crate::process::fbm::Fbm::sample_par)'s own doc).
/// None of this is a defect to fix — it is a weaker, explicitly measured
/// and documented contract for those backends specifically, standing
/// alongside, not contradicting, the unconditional guarantee below.
///
/// Same seed + same `m` ⇒ bit-identical output on any machine, any
/// thread-pool size, and any chunking of `m` (chunks need not each hold
/// exactly one path for them to be mutually independent), for every
/// process in this crate on its default `Cpu` backend.
/// [`Unseeded`](stochastic_rs_core::simd_rng::Unseeded) processes still draw
/// fresh randomness on every call.
///
/// Implementor footgun: the default `sample()` routes through `sampler()`,
/// so a sampler must never call back into `ProcessExt::sample` of the same
/// process unless that process overrides `sample` with a real body.
///
/// ## `Clone` semantics
///
/// Every process type in this crate that implements `Clone` today does so
/// via a plain `#[derive(Clone)]` over a `seed: S` field — there is no
/// per-type override. For
/// [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic) seeds
/// that derive clones `seed` field-wise, which resolves to
/// `Deterministic::clone()`: a byte-for-byte copy of the seed's *current*
/// counter into a fresh, independent `AtomicU64` (see that type's own doc —
/// "Cloning snapshots the current state"). So whole-struct `Clone`
/// **snapshots** the seed rather than forking it:
///
/// ```
/// use stochastic_rs_core::simd_rng::Deterministic;
/// use stochastic_rs_stochastic::diffusion::ou::Ou;
/// use stochastic_rs_stochastic::traits::ProcessExt;
///
/// let a = Ou::<f64, _>::new(0.5, 0.02, 0.1, 32, Some(0.03), Some(1.0), Deterministic::new(42));
/// let b = a.clone();
/// // `b` snapshots `a`'s seed exactly as it stood at `.clone()`, so sampled
/// // immediately afterward — before either side draws anything else — the
/// // two agree bit-for-bit.
/// assert_eq!(a.sample(), b.sample());
/// ```
///
/// [`Unseeded`](stochastic_rs_core::simd_rng::Unseeded) makes the
/// distinction moot: it carries no state, so snapshot and fork coincide,
/// and `.sample()` still draws fresh randomness on every call regardless of
/// how many times the process was cloned beforehand.
///
/// This is a deliberate choice, and it intentionally diverges from
/// `stochastic-rs-distributions`, where `Clone` on a distribution (e.g.
/// [`SimdNormal`](stochastic_rs_distributions::normal::SimdNormal))
/// re-seeds independently by design ("cloning a stochastic source means
/// 'give me an independent stream'"). The two crates answer different
/// questions: a distribution is typically cloned to obtain an unrelated
/// sampler, while a process is typically cloned to answer "same model, one
/// parameter changed" — `let bumped = base.clone(); bumped.kappa += h;` —
/// which only isolates `h`'s effect if `bumped` and `base` share the same
/// underlying noise. That is the common-random-numbers technique behind
/// finite-difference Greeks ([`MalliavinExt`](crate::traits::MalliavinExt))
/// and bump-and-reprice sensitivity analysis; forking on clone would
/// silently replace every such comparison with uncorrelated Monte Carlo
/// noise instead of isolating the bumped parameter.
///
/// **Tradeoff accepted:** a caller who clones a process purely to obtain a
/// second, *independent* simulation stream — not to bump a parameter — gets
/// the identical-path behaviour above if they sample both sides before
/// either has drawn anything else, which is easy to trip over silently. For
/// an independent stream, construct a fresh seed instead of cloning
/// (`Deterministic::new(new_seed)`, or `Unseeded`), or call
/// [`reseed`](stochastic_rs_core::simd_rng::SeedExt::reseed) on the clone's
/// `seed` field before sampling it.
///
/// Not to be confused with the *internal* rule in "Reproducibility
/// requirement on implementors" above: that rule constrains how one
/// process's own `sampler()` builds its per-call/per-chunk basis
/// (`derive()`, never `clone()`, on `self.seed`, from inside a single
/// `&self` call). This section is about the outer, whole-struct `Clone` a
/// caller invokes once, from outside, before ever calling `sample()` — the
/// two are unrelated, and this section's guarantee holds no matter how any
/// individual type's `sampler()` is implemented.
pub trait ProcessExt<T: FloatExt>: Send + Sync {
  type Output: Send;

  /// Reusable sampling state. Implementation detail of the `sample*` methods,
  /// not part of the public surface.
  #[doc(hidden)]
  type Sampler<'a>: PathSampler<T, Output = Self::Output>
  where
    Self: 'a;

  /// Constructs the reusable sampling state. Implementation detail behind
  /// [`sample`](Self::sample) / [`sample_map`](Self::sample_map).
  #[doc(hidden)]
  fn sampler(&self) -> Self::Sampler<'_>;

  /// Advances this process's own `Deterministic` seed state by one tick,
  /// discarding the returned value — called purely for the side effect.
  /// Called from [`chunked_samplers`](Self::chunked_samplers) (once per
  /// chunk, before that chunk's `sampler()`) and from
  /// [`sample`](Self::sample) (once per call, after sampling).
  ///
  /// The default is a no-op, correct for every process whose `sampler()`
  /// derives its own basis at construction (see the trait-level
  /// "Reproducibility requirement on implementors" section — this is now
  /// the required shape, so the default covers almost every process in the
  /// crate); an *additional* advance here would be harmless but redundant
  /// for those. **Override this** for the narrow remaining case: a
  /// `sampler()` that clones the seed (`self.seed.clone()`, a
  /// non-advancing snapshot per `SeedExt`'s design) because the clone feeds
  /// a persistent engine reused across a whole chunk rather than being
  /// re-derived per path — see
  /// [`CirPlusPlus`](crate::interest::cir_pp::CirPlusPlus)'s override,
  /// `fn advance_chunk_seed(&self) { self.seed.seed_value(); }` — so each
  /// chunk's clone snapshots a distinct state instead of every chunk
  /// replaying the same one, and repeated top-level `sample()` calls
  /// advance instead of replaying the first path forever.
  #[doc(hidden)]
  fn advance_chunk_seed(&self) {}

  /// Builds one sampler per chunk, paired with that chunk's path count,
  /// **sequentially on the calling thread, before any chunk reaches
  /// rayon** — each chunk's sampler is only *distinctly* seeded (the
  /// "pre-seeded" this method's implementors must deliver) if the process
  /// satisfies the trait-level reproducibility requirement: most do so via
  /// `sampler()` alone; the rest must override
  /// [`advance_chunk_seed`](Self::advance_chunk_seed), which this method
  /// calls immediately before each [`sampler()`](Self::sampler) call, in
  /// this fixed sequential order. That sequencing is what fixes the order
  /// in which a `Deterministic` process's shared seed state is consumed,
  /// independent of how rayon later schedules the chunks — implementation
  /// detail behind [`sample_par`](Self::sample_par) /
  /// [`sample_map`](Self::sample_map).
  #[doc(hidden)]
  fn chunked_samplers(&self, m: usize) -> Vec<(Self::Sampler<'_>, usize)> {
    let chunks = chunk_count(m);
    chunk_lens(m, chunks)
      .map(|len| {
        self.advance_chunk_seed();
        (self.sampler(), len)
      })
      .collect()
  }

  /// A single sampled path.
  ///
  /// Ticks [`advance_chunk_seed`](Self::advance_chunk_seed) *after* sampling
  /// (not before): a clone-based `sampler()` (see that method's docs) reads
  /// `self.seed`'s *current* state to build the sampler, so ticking first
  /// would make this call's own basis skip a state the caller never
  /// consumed — ticking after instead advances `self.seed` by exactly the
  /// same one step the sampler's own internal derive just consumed from its
  /// clone, so the *next* independent `sample()` call picks up where this
  /// one left off (matching a process whose `sampler()` advances directly,
  /// which this is a no-op for). This is what keeps repeated top-level
  /// `sample()` calls on one process advancing, for every process in the
  /// crate, not only the ones with an owned, per-call-advancing sampler.
  fn sample(&self) -> Self::Output {
    let out = self.sampler().sample();
    self.advance_chunk_seed();
    out
  }

  /// Maps `f` over `m` independently sampled paths, reusing one sampler and
  /// one output buffer per chunk (no per-path allocation or RNG re-init).
  /// This is the parallel primitive.
  ///
  /// **Reproducibility.** Every process in this crate satisfies the
  /// trait-level "Reproducibility requirement on implementors" (see
  /// [`ProcessExt`]'s own docs — no exceptions, full or partial), so same
  /// seed + same `m` ⇒ bit-identical output, on any machine and under any
  /// rayon thread-pool size. `Unseeded` processes still draw fresh
  /// randomness on every call.
  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Self::Output) -> R + Sync) -> Vec<R> {
    if m == 0 {
      return Vec::new();
    }
    if m == 1 {
      return vec![f(&self.sample())];
    }
    self
      .chunked_samplers(m)
      .into_par_iter()
      .map(|(mut sampler, len)| {
        let mut slot: Option<Self::Output> = None;
        (0..len)
          .map(|_| {
            if let Some(buf) = slot.as_mut() {
              sampler.sample_into(buf);
              return f(buf);
            }
            // First path in this chunk: sample fresh (no wasted draw) and
            // keep the buffer to reuse for the rest of the chunk.
            let buf = sampler.sample();
            let r = f(&buf);
            slot = Some(buf);
            r
          })
          .collect::<Vec<_>>()
      })
      // Chunks run on rayon, but `Vec::into_par_iter()` → `.map()` is an
      // `IndexedParallelIterator`, so `.collect()` restores chunk order
      // regardless of completion order; flattening then reproduces the
      // exact global path order (chunk 0's paths, then chunk 1's, ...).
      .collect::<Vec<_>>()
      .into_iter()
      .flatten()
      .collect()
  }

  /// `m` independently sampled paths, kept. Like [`sample_map`](Self::sample_map)
  /// it reuses one sampler per chunk, but allocates a fresh owned path each
  /// step — cheaper than mapping then cloning when every path is wanted.
  ///
  /// **Reproducibility.** Same guarantee as [`sample_map`](Self::sample_map)
  /// above.
  fn sample_par(&self, m: usize) -> Vec<Self::Output> {
    if m == 0 {
      return Vec::new();
    }
    if m == 1 {
      return vec![self.sample()];
    }
    self
      .chunked_samplers(m)
      .into_par_iter()
      .map(|(mut sampler, len)| (0..len).map(|_| sampler.sample()).collect::<Vec<_>>())
      // Same order-preserving collect-then-flatten as `sample_map` above.
      .collect::<Vec<_>>()
      .into_iter()
      .flatten()
      .collect()
  }
}

/// Marker for processes whose [`ProcessExt::sample`] returns a single
/// 1D trajectory `Array1<T>`.
///
/// Auto-implemented via a blanket impl for any `P: ProcessExt<T, Output = Array1<T>>`,
/// so the user only needs to query the marker — no manual `impl` lines on each
/// process struct. Use this in generic code that should only operate on single-path
/// processes (e.g. `Bm`, `Ou`, `Gbm`, `Vasicek`).
///
/// ```
/// use stochastic_rs_stochastic::process::bm::Bm;
/// use stochastic_rs_stochastic::simd_rng::Deterministic;
/// use stochastic_rs_stochastic::traits::{FloatExt, OneDimensional, ProcessExt};
///
/// fn last_value<T: FloatExt, P: OneDimensional<T>>(p: &P) -> T {
///   *p.sample().last().unwrap()
/// }
///
/// // `Bm` has `Output = Array1<T>`, so the blanket impl gives it
/// // `OneDimensional<T>` for free — no manual `impl` needed.
/// let bm = Bm::<f64, _>::new(64, Some(1.0), Deterministic::new(1));
/// let v = last_value(&bm);
/// assert!(v.is_finite());
/// ```
pub trait OneDimensional<T: FloatExt>: ProcessExt<T, Output = Array1<T>> {}

impl<T: FloatExt, P> OneDimensional<T> for P where P: ProcessExt<T, Output = Array1<T>> {}

/// Marker for processes whose [`ProcessExt::sample`] returns `N` aligned
/// 1D trajectories `[Array1<T>; N]`.
///
/// Auto-implemented for any `P: ProcessExt<T, Output = [Array1<T>; N]>`.
/// Stochastic-volatility models (`Heston`, `Bergomi`, `Sabr`, `RBergomi`)
/// use `N = 2` (asset + variance); 3-state models (`HestonStochCorr`,
/// `DoubleHeston`, `Hjm`) use `N = 3`.
pub trait MultiDimensional<T: FloatExt, const N: usize>:
  ProcessExt<T, Output = [Array1<T>; N]>
{
}

impl<T: FloatExt, P, const N: usize> MultiDimensional<T, N> for P where
  P: ProcessExt<T, Output = [Array1<T>; N]>
{
}

/// Convenience marker for the common 2-state case `[Array1<T>; 2]`.
///
/// Subtrait of [`MultiDimensional<T, 2>`]. Useful for asset-plus-variance
/// stochastic-vol models like `Heston`, `Bergomi`, `Sabr`.
pub trait TwoDimensional<T: FloatExt>: MultiDimensional<T, 2> {}

impl<T: FloatExt, P> TwoDimensional<T> for P where P: MultiDimensional<T, 2> {}

/// Marker for processes whose [`ProcessExt::sample`] returns an `Array2<T>`
/// matrix — a discretised curve or sheet rather than a single path.
///
/// Auto-implemented for any `P: ProcessExt<T, Output = Array2<T>>`. Used by
/// interest-rate term-structure models (`Hjm`-with-tenors, `WuZhangD`),
/// stochastic-sheet processes (`Fbs`), and the parallel-rate primitive
/// `Bgm` (which despite its name is not a coupled BGM/LMM — see its module
/// doc).
pub trait CurveOutput<T: FloatExt>: ProcessExt<T, Output = Array2<T>> {}

impl<T: FloatExt, P> CurveOutput<T> for P where P: ProcessExt<T, Output = Array2<T>> {}

/// Marker for processes whose [`ProcessExt::sample`] returns a runtime-sized
/// collection of 1D trajectories `Vec<Array1<T>>`.
///
/// Auto-implemented for any `P: ProcessExt<T, Output = Vec<Array1<T>>>`.
/// Used when the dimensionality `D` is determined by a runtime parameter and
/// each component carries its own (possibly variable-length) trace —
/// e.g. [`crate::process::multivariate_hawkes::MultivariateHawkes`], whose
/// per-component event-time vectors have process-dependent lengths.
pub trait VariableDimensional<T: FloatExt>: ProcessExt<T, Output = Vec<Array1<T>>> {}

impl<T: FloatExt, P> VariableDimensional<T> for P where P: ProcessExt<T, Output = Vec<Array1<T>>> {}

/// Marker for processes whose [`ProcessExt::sample`] returns a complex-valued
/// 1D trajectory `Array1<Complex<T>>`.
///
/// Auto-implemented for any `P: ProcessExt<T, Output = Array1<Complex<T>>>`.
/// Used by complex-state diffusions such as
/// [`crate::diffusion::cfou::Cfou`], where the joint dynamics of two
/// real OU components are expressed as a single complex Ornstein-Uhlenbeck
/// `Z_t = X_1(t) + i X_2(t)`.
pub trait ComplexPathOutput<T: FloatExt>:
  ProcessExt<T, Output = ndarray::Array1<num_complex::Complex<T>>>
{
}

impl<T: FloatExt, P> ComplexPathOutput<T> for P where
  P: ProcessExt<T, Output = ndarray::Array1<num_complex::Complex<T>>>
{
}
