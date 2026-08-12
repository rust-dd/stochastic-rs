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
fn chunk_count(m: usize) -> usize {
  m.min(MAX_CHUNKS)
}

/// Splits `m` into `chunks` contiguous run lengths, as even as possible (the
/// first `m % chunks` chunks get one extra path), yielded in chunk order.
///
/// `chunks == 0` only ever arises from `chunk_count(0)`; the `checked_div`/
/// `checked_rem` fall back to `0` there (rather than relying on every caller
/// to check `m` first) so this function stays total instead of panicking on
/// the `m / 0` that a plain division would perform.
fn chunk_lens(m: usize, chunks: usize) -> impl Iterator<Item = usize> {
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
/// any chunk reaches rayon. [`sample_map`] folds over each chunk's paths
/// reusing a single output buffer; [`sample_par`] keeps every path,
/// allocating each fresh (no buffer reuse, no clone).
///
/// ### Reproducibility requirement on implementors
///
/// Sequential chunk construction only produces bit-identical, thread-count-
/// independent output if **every chunk's sampler draws from a distinct seed
/// basis**. There are two ways for an implementor to guarantee that, and a
/// process must use one of them:
///
/// - `sampler()` itself advances the process's `Deterministic` seed state at
///   construction time (e.g. it builds a `SimdNormal`/`SimdPoisson`/… from
///   `&self.seed` or `&self.seed.derive()`, or owns a chunk-specific `S`
///   captured via `self.seed.derive()`) — true for most processes in this
///   crate, and nothing further is required.
/// - `sampler()` instead *clones* the seed (`self.seed.clone()`), which is
///   `SeedExt`'s designed inverse of advancing — a `Deterministic::clone()`
///   is a non-advancing snapshot. Such a process must override
///   [`advance_chunk_seed`](Self::advance_chunk_seed) to advance the shared
///   state itself before `sampler()` runs, or every chunk clones the same
///   snapshot and `sample_par(m)` degenerates to at most `MAX_CHUNKS`
///   distinct paths repeated, independent of `m`.
///
/// A process whose `sampler()` reads `&self.seed` *lazily*, per path, from
/// inside the returned sampler (rather than once at construction) satisfies
/// neither shape — every chunk's sampler shares live access to the same
/// atomic, so concurrent chunks race on it during the parallel region
/// itself, which no amount of pre-parallel sequencing can fix. That shape
/// must be rewritten to capture an owned `self.seed.clone()` at `sampler()`
/// construction (not `.derive()` — cloning first and deriving from the
/// clone inside the existing per-path code, unchanged, reproduces the exact
/// value a direct `self.seed.derive()` there would have; deriving *both* at
/// construction *and* again per path shifts the value for no reason),
/// converting it to the second bullet above — an `advance_chunk_seed`
/// override is required alongside the rewrite.
///
/// Two in-tree processes cannot satisfy either shape because their sampled
/// randomness does not derive from `self.seed` **at all**, by pre-existing
/// design predating this requirement: [`Bates1996`](crate::jump::bates::Bates1996)
/// (diffusion hard-wires an `Unseeded` correlated-Gaussian source; the jump
/// component reads its own driver's seed field directly, bypassing this
/// trait) and [`RoughHeston`](crate::volatility::fheston::RoughHeston) (its
/// correlated-Gaussian source is documented as ignoring `self.seed`
/// entirely). Their `sample`/`sample_par`/`sample_map` are not seed-
/// reproducible at all — not even serially, not even at `m == 1` — so the
/// reproducibility guarantee below does not apply to them; see MIGRATION.md.
///
/// Same seed + same `m` ⇒ bit-identical output on any machine and any
/// thread-pool size for every process satisfying the requirement above;
/// [`Unseeded`](stochastic_rs_core::simd_rng::Unseeded) processes still draw
/// fresh randomness on every call.
///
/// Implementor footgun: the default `sample()` routes through `sampler()`,
/// so a sampler must never call back into `ProcessExt::sample` of the same
/// process unless that process overrides `sample` with a real body.
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
  /// The default is a no-op, correct for any process whose `sampler()`
  /// itself advances the seed at construction (the common case — see the
  /// trait-level "Reproducibility requirement on implementors" section);
  /// an *additional* advance here would be harmless but redundant for those.
  /// **Override this** for a process whose `sampler()` instead clones the
  /// seed (`self.seed.clone()`, a non-advancing snapshot per `SeedExt`'s
  /// design) — e.g. `fn advance_chunk_seed(&self) { self.seed.seed_value(); }`
  /// — so each chunk's clone snapshots a distinct state instead of every
  /// chunk replaying the same one, and repeated top-level `sample()` calls
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
  /// **Reproducibility.** For a process satisfying the trait-level
  /// "Reproducibility requirement on implementors" (see [`ProcessExt`]'s
  /// own docs — true for every process in this crate except the two named
  /// exceptions there), same seed + same `m` ⇒ bit-identical output, on any
  /// machine and under any rayon thread-pool size. `Unseeded` processes
  /// still draw fresh randomness on every call.
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
  /// **Reproducibility.** Same guarantee and the same caveat as
  /// [`sample_map`](Self::sample_map) above.
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
/// ```ignore
/// fn last_value<T: FloatExt, P: OneDimensional<T>>(p: &P) -> T {
///   *p.sample().last().unwrap()
/// }
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
