//! `ProcessExt` and dimensional output markers.

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::parallel::prelude::*;
use stochastic_rs_distributions::traits::FloatExt;

use super::sampler::PathSampler;
use crate::device::DeviceError;

mod chunks;
mod markers;

pub(crate) use chunks::chunk_count;
pub(crate) use chunks::chunk_lens;
pub(crate) use chunks::sample_map_chunked;
pub(crate) use chunks::sample_par_chunked;
pub(crate) use chunks::try_sample_par_chunked;
pub use markers::ComplexPathOutput;
pub use markers::CurveOutput;
pub use markers::MultiDimensional;
pub use markers::OneDimensional;
pub use markers::TwoDimensional;
pub use markers::VariableDimensional;

/// Stochastic process simulation trait.
///
/// Each process exposes `sample()` returning a [`Self::Output`] and
/// `sample_par(m)` returning `m` independent samples via Rayon.
///
/// ## Time-horizon (`t`) convention
///
/// Most process structs hold `t: Option<T>`. When `t.is_none()`, implementations
/// fall back to `T::one()` (i.e. one unit of time, conventionally one year for
/// finance models). This is the convention across the workspace; do **not** rely
/// on it implicitly for interest-rate or
/// volatility models where the horizon meaningfully drives parameter scaling
/// (Vasicek, CIR, HJM, Heston, Bergomi). For those, set `t` explicitly.
///
/// ## Backend selection
///
/// `process.on::<B>()` re-types a process to a device's default handle and `with_backend(handle)`
/// to an explicit one; every process has both, and a device marker exists only where its feature
/// is compiled (Metal on Apple silicon only). A configuration a device cannot run samples on the
/// host, bit-identically to [`Cpu`](crate::device::Cpu):
/// [`device_fallback`](Self::device_fallback) names the reason.
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
/// ### What reproducibility means here
///
/// Three questions get confused, and only the first is a promise.
///
/// **Same seed, same `m`, same backend, same build.** Bit-identical, every
/// run, on any machine, under any rayon thread-pool size. This is the
/// guarantee, and it is what the crate's reproducibility suite tests.
///
/// **Same seed, different backend.** *Never* identical, by construction. The
/// host draws from this crate's SIMD ziggurat; a device kernel draws its own
/// normals from Philox counters, and the fractional pipelines transform
/// their noise in a different order again. What the two share is the *law*,
/// which is what the device-law suite compares — the terminal mean, the
/// spread, the covariance, each against the host's own. A program that
/// depends on a host path and a device path being the same numbers was never
/// going to work; one that depends on them having the same distribution will.
///
/// **Same seed, different version of this crate.** Not promised. A sampler's
/// stream is an implementation detail, and two things move it: a correctness
/// fix — where the old numbers were wrong, so they had to — and a
/// performance change that reorders how draws are assigned to paths. The
/// second kind is called out in the release notes with the backend and the
/// processes it touches, so a stored result can be re-blessed deliberately
/// rather than discovered by a failing diff. Within a release, nothing moves.
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
/// feeds a *persistent* engine (e.g. a `Seeded` normal stream) built once per
/// chunk and reused across every path in that chunk via the engine's own
/// internal advancement, never re-consulting the `Deterministic`-level seed
/// per path — see [`CirPlusPlus`](crate::interest::cir_pp::CirPlusPlus).
/// Overriding it to advance the shared state before each chunk's
/// `sampler()` call gives each such clone a distinct starting point.
///
/// **No process in this crate is an exception**: every implementor draws all of its randomness,
/// jumps included, from `self.seed`.
///
/// **This guarantee is enforced**: `tests/reproducibility_all_processes.rs` checks every concrete
/// implementor under `sample` and under `sample_par` at one and at several paths per chunk.
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
/// [`FgnBackend`](crate::device::FgnBackend)'s own doc for the full per-backend
/// table and `tests/deterministic_parallelism_accelerate.rs` for the
/// measurement. GPU backends (`Cuda`/`Metal`) are
/// excluded from this guarantee entirely, deliberately: each draws one
/// value from `self.seed.rng()` per batch call and keys the kernels'
/// Philox-4x32-10 on it, so output is a function of
/// the pinned seed but *not* of host thread-pool size, yet cross-run
/// bit-identity across GPU driver versions, vendors, or repeated runs on
/// the same device is untested and not promised.
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
/// This is a deliberate choice, the one a seeded stream ([`Seeded`](stochastic_rs_distributions::Seeded))
/// makes as well: a process is typically cloned to answer "same model, one
/// parameter changed" — `let bumped = base.clone(); bumped.kappa += h;` —
/// which only isolates `h`'s effect if `bumped` and `base` share the same
/// underlying noise. That is the common-random-numbers technique behind
/// finite-difference Greeks and bump-and-reprice sensitivity analysis; forking on clone would
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
pub trait ProcessExt<T: FloatExt>: Send + Sync + crate::traits::Sealed {
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
  /// `fn advance_chunk_seed(&self) { self.seed.next_seed(); }` — so each
  /// chunk's clone snapshots a distinct state instead of every chunk
  /// replaying the same one, and repeated top-level `sample()` calls
  /// advance instead of replaying the first path forever.
  #[doc(hidden)]
  fn advance_chunk_seed(&self) {}

  /// Why this configuration cannot run on a device kernel, or `None` when it
  /// can — a grid past a per-path array, a dimension past the state slots, a
  /// coefficient that is a closure, a mode with no grid.
  ///
  /// The reason is what [`device_ready`](Self::device_ready) leaves out, and
  /// leaving it out is a trap: a fallback costs an order of magnitude in
  /// speed and says nothing, so a caller who cares about running on the
  /// device has to be able to print the cause. A process that the kernels
  /// carry whole answers `None`, which is the default.
  ///
  /// ```ignore
  /// if let Some(why) = process.device_fallback() {
  ///   eprintln!("sampling on the host: {why}");
  /// }
  /// ```
  fn device_fallback(&self) -> Option<&'static str> {
    None
  }

  /// Whether this configuration runs on a device backend. `true` for a
  /// process the device kernels carry whole; `false` where the configuration
  /// exceeds what they carry, which is exactly when
  /// [`device_fallback`](Self::device_fallback) names a reason.
  /// A `false` never fails: on any backend the process then samples on the
  /// host through its own sampler, bit-identically to the `Cpu` build. The
  /// question is about the configuration, not the handle, so the host
  /// backends answer it the same way a device does.
  fn device_ready(&self) -> bool {
    self.device_fallback().is_none()
  }

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
    sample_map_chunked(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<Self::Output> {
    sample_par_chunked(self, m)
  }

  /// [`sample_map`](Self::sample_map) where the callback borrows the path
  /// instead of owning it — for the processes whose sample is a single row.
  ///
  /// The two produce the same values; what differs is who allocates. A
  /// device batch arrives as one `m × n` block, and handing an owned
  /// `Array1` to the callback means copying every row out of it first. A
  /// view is that same memory, so the block is read where it lies and the
  /// batch is traversed once — worth 1.4-2.1x on Metal, where the block is
  /// the buffer the kernel wrote and the row copy is then the only copy in
  /// the call. On the host, where a path is already an owned array, the view
  /// costs nothing either way.
  ///
  /// Most callbacks compile unchanged — `ArrayView1` indexes, iterates and
  /// reduces like the array it borrows — so this is the one to reach for
  /// when the result is a number per path rather than the path itself.
  fn sample_map_view<R: Send>(&self, m: usize, f: impl Fn(ArrayView1<T>) -> R + Sync) -> Vec<R>
  where
    Self: ProcessExt<T, Output = Array1<T>> + Sized,
  {
    self.sample_map(m, |path| f(path.view()))
  }

  /// [`sample`](Self::sample), reporting a device failure as a
  /// [`DeviceError`] instead of panicking. On the host devices it cannot
  /// fail and the path is the one `sample` would have produced.
  fn try_sample(&self) -> Result<Self::Output, DeviceError> {
    let out = self.sampler().try_sample();
    self.advance_chunk_seed();
    out
  }

  /// [`sample_par`](Self::sample_par), reporting a device failure as a
  /// [`DeviceError`] instead of panicking. On the host devices it cannot
  /// fail and the paths are bit-identical to `sample_par`'s.
  fn try_sample_par(&self, m: usize) -> Result<Vec<Self::Output>, DeviceError> {
    try_sample_par_chunked(self, m)
  }
}
