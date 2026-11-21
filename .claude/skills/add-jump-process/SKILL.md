---
name: add-jump-process
description: How to add a jump-diffusion / Lévy / compound-Poisson process to stochastic-rs-stochastic. Invoke for Merton-jump, Kou-jump, Bates-style models, or for layering jumps onto an existing diffusion (GBM → MJD, Heston → Bates).
---

# Add jump process — stochastic-rs-stochastic

A jump process in `stochastic-rs-stochastic` is parameterised by a
**generic** jump-size distribution `D: rand::distr::Distribution<T> +
Send + Sync`, keeping the jump kernel orthogonal from the diffusion.
Its jumps are a private `crate::process::cpoisson::JumpDriver<D, S>`:
the law and the seed the arrivals derive from, at the process's own
`lambda`. It holds no `Poisson`, so a zero rate draws zero increments.

Read `add-diffusion-process` first: a jump process is a diffusion
process that additionally owns a `JumpDriver` field. The
`ProcessExt` contract, the `sampler()` / `PathSampler` split, and the
`seed`-last constructor convention are identical and are documented
there, not repeated here.

## 1. The pattern: a private `JumpDriver<D, S>`

```rust
// stochastic-rs-stochastic/src/jump/merton.rs (reference)

use ndarray::Array1;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::process::cpoisson::JumpDriver;
use crate::traits::FloatExt;

#[derive(Clone)]
pub struct Merton<T, D, S: SeedExt = Unseeded>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub alpha: T,          // log-price drift μ
  pub sigma: T,          // Brownian diffusion scale
  pub lambda: T,         // jump intensity λ — the one rate
  pub theta: T,          // jump-size compensator κ, NOT a mean-reversion level
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  jumps: JumpDriver<D, S>,
  pub seed: S,
}

impl<T, D, S: SeedExt> Merton<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    alpha: T, sigma: T, lambda: T, theta: T,
    jump_dist: D,
    n: usize, x0: Option<T>, t: Option<T>,
    seed: S,                                   // <- seed is LAST
  ) -> Self {
    // asserts: every float finite, `lambda >= T::zero()`, `n >= 2`, ...
    let jumps = JumpDriver::new(jump_dist, &seed); // the arrivals' seed: a child of `seed`
    Self { alpha, sigma, lambda, theta, n, x0, t, jumps, seed }
  }
}
```

Three constructor facts:

- **`seed: S` is the last parameter**, and the caller supplies the jump
  *distribution* and *intensity* directly. There is no `seeded(...)`
  constructor — that shape is pre-3.0 and gone.
- **The jump driver is built internally** by `JumpDriver::new(law, &seed)`,
  which takes `seed.clone().derive()`. Do not ask the caller to thread a
  third, independent seed.
- **`lambda >= 0`.** A zero rate is the diffusion alone: `grid_increments`
  draws nothing where `lambda·dt` is zero.

### One rate, one law

`lambda` is the only rate: the sampler and the engine's `jump_intensity()`
both read it, and nothing mirrors it. The setters are `with_lambda(λ)`
(asserting what `new` asserts) and `with_jump_law(law)` (keeping the jump
seed); `jump_law()` reads the law back.

## 2. The sample step

You implement `ProcessExt::sampler()`, returning a `PathSampler`. The
sampler owns its diffusion noise source and an owned, chunk-local jump
seed; it must **never** borrow the driver's seed, or every chunk races
on the same shared atomic during the parallel region.

```rust
impl<T, D, S: SeedExt> crate::traits::Sealed for Merton<T, D, S> where T: FloatExt, D: Distribution<T> + Send + Sync {}

impl<T, D, S: SeedExt> ProcessExt<T> for Merton<T, D, S>
where T: FloatExt, D: Distribution<T> + Send + Sync
{
  type Output = Array1<T>;
  type Sampler<'s> = MertonSampler<'s, T, D, S> where Self: 's;

  fn sampler(&self) -> MertonSampler<'_, T, D, S> {
    let dt = if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    };
    // Lévy compensator folded into the deterministic drift, once:
    let drift_dt = (self.alpha
      - self.sigma.powf(T::from_usize(2).unwrap()) / T::from_usize(2).unwrap()
      - self.lambda * self.theta) * dt;

    MertonSampler {
      n: self.n,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      drift_dt,
      jump_distribution: &self.jumps.law,   // borrow: read-only params
      lambda: self.lambda,
      jump_seed: self.jumps.seed.derive(), // own: chunk-local basis
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(&self.seed),
    }
  }
}
```

and the path fill itself:

```rust
impl<T, D, S: SeedExt> crate::traits::Sealed for MertonSampler<'_, T, D, S> where T: FloatExt, D: Distribution<T> + Send + Sync {}

impl<T, D, S: SeedExt> PathSampler<T> for MertonSampler<'_, T, D, S>
where T: FloatExt, D: Distribution<T> + Send + Sync
{
  type Output = Array1<T>;
  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(out.as_slice_mut().expect("Merton output must be contiguous"));
  }
  fn sample(&mut self) -> Array1<T> {
    array1_from_fill(self.n, |out| self.fill_path(out))
  }
}

// inherent helper on the sampler
fn fill_path(&mut self, out: &mut [T]) {
  if out.is_empty() { return; }

  let jump_increments = crate::process::cpoisson::grid_increments(
    self.jump_distribution, self.lambda, &self.jump_seed, out.len(), self.dt,
  );
  let mut gn = Array1::<T>::zeros(out.len() - 1);
  if let Some(gn_slice) = gn.as_slice_mut() {
    self.normal.fill_slice(gn_slice);
  }

  out[0] = self.x0;
  for i in 1..out.len() {
    out[i] = out[i - 1] + self.drift_dt + self.sigma * gn[i - 1] + jump_increments[i];
  }
}
```

Notes on the jump-increment call:

- `crate::process::cpoisson::grid_increments(distribution, lambda,
  seed, n, dt)` is a `pub(crate)` free function — available to any
  process inside this crate, not to downstream users.
- The public equivalents on the struct are
  `CompoundPoisson::sample_grid_increments(n, dt)` and
  `sample_grid_relative_increments(n, dt)` (the latter compounds
  multiplicatively — use it for a relative-return jump term).
- There is **no** `sample_increments` and **no** `sample_arrival_times`.
  Arrival times come from driving `CompoundPoisson` itself: its
  `ProcessExt::Output` is `process::CompoundPaths { arrivals, cumulative, jumps }`.
- The jump array is indexed `jump_increments[i]`, not `[i - 1]` — index
  0 is the (zero) increment at the path's initial point.

**Compensator.** Risk-neutral pricing requires `E[exp(X_t)]` to grow at
`r - q`, so the deterministic drift must absorb the expected jump
contribution. In `Merton` this is the `- self.lambda * self.theta` term,
computed **once** in `sampler()` and baked into `drift_dt` — not
recomputed inside the per-step loop. Forgetting it is the most common
silent-correctness bug in jump-process implementations.

## 3. Choosing `D` — this is where `dev-rules` §7a binds

`D` must be `Distribution<T> + Send + Sync`.

A jump-size law is any scalar continuous `Simd*` type except `SimdNonCentralChiSquared` (per-draw noncentrality): it is
`Send + Sync`, and its `Distribution::sample(&mut rng)` draws from the generator the process holds. The sampler's own
stream (`normal: Seeded<SimdNormal<T>>` above) is the other slot; do not confuse the two.

```rust
// Merton's own Default: the canonical choice for the jump slot
impl<T: FloatExt> Default for Merton<T, SimdNormal<T>, Unseeded> {
  fn default() -> Self {
    Self::new(/* ... */, SimdNormal::new(T::zero(), T::from_f64_fast(0.1)), /* ... */)
  }
}
```

The device engine recognises `SimdNormal` and `SimdExp` only. Notably absent is a signed asymmetric double-exponential,
the actual Kou (2002) jump law: `Kou` ships no `Default` for that reason; do not substitute `SimdNormal`. There is no
`SimdNig` either: the Normal-Inverse-Gaussian law is `SimdNormalInverseGauss`, itself a valid jump law.

The trait comes from `rand::distr::Distribution` (`rand_distr` is not a dependency of this crate) —
our own types implement it, and it is how `.sample()` resolves. Per `dev-rules` §7a the concrete
`rand_distr` distributions stay out of library code.

## 4. Construction-time parameter validation

Validate in `new(...)`, never in `sample()`. `Poisson::new` already
checks its `Horizon`; add your own asserts for the model's own
invariants (every float finite, `sigma > 0`, `lambda >= 0`, `n >= 2`) in
dev-rules §14's form, which a `#[should_panic(expected = "…")]` test can pin.

Note that `Merton` and `Kou` as shipped do **not** carry `r` / `r_f` /
`b` cost-of-carry fields — those live on the pricers in
`stochastic-rs-quant`, not on the simulation processes. If you are
looking for the `b == r - r_f` consistency check, it belongs there.

## 5. Characteristic function

If the corresponding pricer (Carr-Madan, Lewis, COS) needs a
characteristic function, the model implements `FourierModelExt`
(`chf(t, xi)` + `cumulants(t)`) on the **quant** side, which
blanket-implements `ModelPricer` through `LewisPricer`, Lewis (2001), less
a Gaussian control. See
`calibration-pattern`. The Lévy-Khintchine triplet must agree with the
SDE actually simulated here:

- drift `= alpha - λ·E[e^J - 1]` (the compensated drift — `theta` is
  this crate's name for that `E[e^J - 1]`-like term),
- diffusion `= sigma` (unchanged by jumps),
- Lévy measure `ν(dx) = λ · f_J(x) dx`.

A ChF that is internally self-consistent but disagrees with the
simulated drift is the failure mode; test them against each other.

## 6. Testing requirements

```rust
#[cfg(test)]
mod tests {
  /// 1. Zero jump intensity → matches the underlying pure diffusion.
  #[test] fn lambda_zero_reduces_to_diffusion() { }

  /// 2. Mean over many paths matches the compensated drift.
  #[test] fn mean_matches_compensated_drift() { }

  /// 3. Seeded determinism — non-negotiable.
  #[test]
  fn seeded_is_deterministic() {
    let a = Merton::new(/* … */, SimdNormal::new(0.0, 0.1), 100, None, None,
                        Deterministic::new(42));
    let b = a.clone();
    assert_eq!(a.sample(), b.sample());
  }

  /// 4. `sample_par(m)` is bit-identical across rayon thread-pool sizes.
  #[test] fn sample_par_is_thread_count_stable() { }
}
```

Test (4) is what the deterministic-parallelism wave exists to protect;
`stochastic-rs-stochastic/tests/reproducibility_all_processes.rs` is the
crate-wide guard and a new process must appear in it. See
`integration-test-writing`.

## 7. Anti-patterns

- **Do not** use `Box<dyn Distribution<T>>`. Use the generic `D`
  parameter — `dev-rules` §3.
- **Do not** invent `seed.advance(...)`, `seed.into_rng()` or a
  `seeded(...)` constructor. `SeedExt`'s full surface is `rng()`,
  `derive()`, `rng_ext::<R>()`, `reseed(s)`, `next_seed()`.
- **Do not** borrow `&self.jumps.seed` into the sampler. Borrow the
  read-only `law`, but **own** a `.seed.derive()`.
- **Do not** recompute the compensator inside the per-step loop. Fold
  it into `drift_dt` in `sampler()`.
- **Do not** forget the compensator entirely. Risk-neutral drift must
  net out the expected jump.

## 8. Reference impls

- `Merton` (`jump/merton.rs`) — GBM + generic jumps; `Default` at
  `D = SimdNormal<T>`; full `with_*` setter set. The template.
- `Kou` (`jump/kou.rs`) — same sampler recursion as Merton, different
  `D`, deliberately no `Default`. Read its type doc for why.
- `Bates` (`jump/bates.rs`) — Heston + generic jumps; `D` is the law of
  the relative jump `Y − 1`, and `LogNormalJump` is Bates (1996)'s own,
  with its compensator as `mean()`; `with_jump_law` keeps `k`, so set it
  to the new law's mean.
- `FbatesSvj` (`volatility/fbates_svj.rs`) — the non-generic fractional
  SVJ variant; it composes path (1) of `add-fractional-process`.
- `LevyDiffusion` (`jump/levy_diffusion.rs`) and `JumpFou`
  (`jump/jump_fou.rs`) — the other drivers of the same shape;
  `JumpFouCustom` (`jump/jump_fou_custom.rs`) takes two laws, the waiting
  times `DT` and the sizes `DS`, and draws its own arrivals.

## Related SKILLs

- `add-diffusion-process` — the `ProcessExt` / `sampler()` contract and
  the diffusion baseline jumps layer onto. Read it first.
- `adding-distribution` — when the jump-size law doesn't exist yet
  (Kou's asymmetric double-exponential is the standing gap).
- `integration-test-writing` — pinned seeds and the all-processes guard.
- `python-bindings` — note that `PyMerton` is **hand-written**, not
  macro-generated, because its jump distribution crosses the PyO3
  boundary as a `CallableDist`.
