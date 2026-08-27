---
name: add-jump-process
description: How to add a jump-diffusion / Lévy / compound-Poisson process to stochastic-rs-stochastic. Invoke for Merton-jump, Kou-jump, Bates-style models, or for layering jumps onto an existing diffusion (GBM → MJD, Heston → Bates).
---

# Add jump process — stochastic-rs-stochastic

A jump process in `stochastic-rs-stochastic` is parameterised by a
**generic** jump-size distribution `D: rand_distr::Distribution<T> +
Send + Sync`, keeping the jump kernel orthogonal from the diffusion.
Compound-Poisson arrivals are handled by
`crate::process::cpoisson::CompoundPoisson`, which the new process
composes as a field.

Read `add-diffusion-process` first: a jump process is a diffusion
process that additionally owns a `CompoundPoisson` field. The
`ProcessExt` contract, the `sampler()` / `PathSampler` split, and the
`seed`-last constructor convention are identical and are documented
there, not repeated here.

## 1. The pattern: composition of `CompoundPoisson<T, D, S>`

```rust
// stochastic-rs-stochastic/src/jump/merton.rs (reference)

use ndarray::Array1;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::scalar::ScalarNormal;

use crate::process::cpoisson::CompoundPoisson;
use crate::process::poisson::Poisson;
use crate::traits::FloatExt;

#[derive(Clone)]
pub struct Merton<T, D, S: SeedExt = Unseeded>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub alpha: T,          // log-price drift μ
  pub sigma: T,          // Brownian diffusion scale
  pub lambda: T,         // jump intensity λ — the single source of truth
  pub theta: T,          // jump-size compensator κ, NOT a mean-reversion level
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub cpoisson: CompoundPoisson<T, D, S>,
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
    let cpoisson = CompoundPoisson::new(
      jump_dist,
      Poisson::new(lambda, Some(n), t, Unseeded),
      seed.clone().derive(),                   // hash-mixed child of the same seed
    );
    Self { alpha, sigma, lambda, theta, n, x0, t, cpoisson, seed }
  }
}
```

Three constructor facts:

- **`seed: S` is the last parameter**, and the caller supplies the jump
  *distribution* and *intensity* directly. There is no `seeded(...)`
  constructor — that shape is pre-3.0 and gone.
- **The jump driver is built internally** from `seed.clone().derive()`.
  Do not ask the caller to thread a third, independent seed.
- **`CompoundPoisson::new(distribution, poisson, seed)`** takes exactly
  three arguments in that order — the jump-size distribution, a
  fully-built `Poisson<T>`, and the seed source. `Poisson::new(lambda,
  n, t_max, seed)` needs one of `n` / `t_max` (it validates).

### `lambda` lives in two places — keep them synced

`sampler()` reads `self.lambda` directly for the arrival rate, **not**
`self.cpoisson.poisson.lambda`. The latter is a cosmetic mirror on the
sampling path but is genuinely live if a caller `.sample()`s the
embedded `CompoundPoisson` standalone. `Merton` keeps them in sync
through `with_lambda` / `with_cpoisson`; a direct `merton.lambda = x`
field assignment is **not** intercepted and desyncs them. If you add
`with_*` setters that can change the intensity, mirror
`resync_cpoisson_poisson`.

## 2. The sample step

You implement `ProcessExt::sampler()`, returning a `PathSampler`. The
sampler owns its diffusion noise source and an owned, chunk-local jump
seed; it must **never** borrow `&self.cpoisson` wholesale, or every
chunk races on the same shared atomic during the parallel region.

```rust
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
      jump_distribution: &self.cpoisson.distribution,   // borrow: read-only params
      lambda: self.lambda,
      jump_seed: self.cpoisson.seed.derive(),           // own: chunk-local basis
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}
```

and the path fill itself:

```rust
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
  `ProcessExt::Output` is `[Array1<T>; 3]`.
- The jump array is indexed `jump_increments[i]`, not `[i - 1]` — index
  0 is the (zero) increment at the path's initial point.

**Compensator.** Risk-neutral pricing requires `E[exp(X_t)]` to grow at
`r - q`, so the deterministic drift must absorb the expected jump
contribution. In `Merton` this is the `- self.lambda * self.theta` term,
computed **once** in `sampler()` and baked into `drift_dt` — not
recomputed inside the per-step loop. Forgetting it is the most common
silent-correctness bug in jump-process implementations.

## 3. Choosing `D` — this is where `dev-rules` §7a binds

`D` must be `Distribution<T> + Send + Sync`. That bound rules out the
`Simd*` distributions: they own an `UnsafeCell` sample buffer and are
`!Sync` by construction. Use the stateless `Scalar*` types from
`stochastic_rs_distributions::scalar`, which sample from the caller's
RNG and exist precisely for this slot:

```rust
// Merton's own Default — the canonical choice for the jump slot
impl<T: FloatExt> Default for Merton<T, ScalarNormal<T>, Unseeded> {
  fn default() -> Self {
    Self::new(/* … */, ScalarNormal::new(T::zero(), T::from_f64_fast(0.1)), /* … */)
  }
}
```

The `Simd*` types still appear in a jump process — but on the
*diffusion* side, where the sampler owns them locally (`normal:
SimdNormal<T>` above). Do not confuse the two slots.

Available `Scalar*` types today are `ScalarNormal` and `ScalarExp`
(`stochastic-rs-distributions/src/scalar.rs`). Notably absent: a signed
**asymmetric double-exponential**, which is the actual Kou (2002) jump
law. `Kou` therefore ships **no `Default`** and its own type doc says so
— a Gaussian `D` would silently hand out Merton-with-Gaussian-jumps
under the `Kou` name. If you need Kou's true law, write the
distribution first (see `adding-distribution`); do not substitute
`ScalarNormal`.

There is no `SimdDoubleExponential` and no `SimdNig`. The
Normal-Inverse-Gaussian distribution is `SimdNormalInverseGauss`
(`stochastic-rs-distributions/src/normal_inverse_gauss.rs`), and being
`Simd*` it is not eligible for the jump slot.

The `rand_distr::Distribution` *trait* import stays — our own types
implement it, and it is how `.sample()` resolves. Per `dev-rules` §7a
only the concrete `rand_distr` distributions are out of library code.

## 4. Construction-time parameter validation

Validate in `new(...)`, never in `sample()`. `Poisson::new` already
calls `validate_n_or_tmax`; add your own asserts for the model's own
invariants (`sigma > 0`, `lambda >= 0`, `n >= 2`) with messages a
`#[should_panic(expected = "…")]` test can pin.

Note that `Merton` and `Kou` as shipped do **not** carry `r` / `r_f` /
`b` cost-of-carry fields — those live on the pricers in
`stochastic-rs-quant`, not on the simulation processes. If you are
looking for the `b == r - r_f` consistency check, it belongs there.

## 5. Characteristic function

If the corresponding pricer (Carr-Madan, Lewis, COS) needs a
characteristic function, the model implements `FourierModelExt`
(`chf(t, xi)` + `cumulants(t)`) on the **quant** side, which
blanket-implements `ModelPricer` via Gil-Pelaez quadrature. See
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
    let a = Merton::new(/* … */, ScalarNormal::new(0.0, 0.1), 100, None, None,
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
- **Do not** put a `Simd*` distribution in the jump slot. It is `!Sync`
  and will not compile; the error is about `Send + Sync`, not about
  jumps, so it reads as confusing. Use `Scalar*`.
- **Do not** invent `seed.advance(...)`, `seed.into_rng()` or a
  `seeded(...)` constructor. `SeedExt`'s full surface is `rng()`,
  `derive()`, `rng_ext::<R>()`, `reseed(s)`, `seed_value()`.
- **Do not** borrow `&self.cpoisson` into the sampler. Borrow the
  read-only `distribution`, but **own** a `.seed.derive()`.
- **Do not** recompute the compensator inside the per-step loop. Fold
  it into `drift_dt` in `sampler()`.
- **Do not** forget the compensator entirely. Risk-neutral drift must
  net out the expected jump.

## 8. Reference impls

- `Merton` (`jump/merton.rs`) — GBM + generic jumps; `Default` at
  `D = ScalarNormal<T>`; full `with_*` setter set. The template.
- `Kou` (`jump/kou.rs`) — same sampler recursion as Merton, different
  `D`, deliberately no `Default`. Read its type doc for why.
- `Bates1996` (`jump/bates.rs`) — Heston + generic jumps; the
  `cpoisson: CompoundPoisson<T, D, S>` widening and its `cgns` caching
  quirk are documented on the type.
- `BatesSvj` / `FBatesSvj` (`volatility/bates_svj.rs`,
  `volatility/fbates_svj.rs`) — non-generic SVJ variants; `FBatesSvj`
  composes path (1) of `add-fractional-process`.
- `LevyDiffusion` (`jump/levy_diffusion.rs`), `JumpFOUCustom`
  (`jump/jump_fou_custom.rs`) — the other two generic-`D` jump slots.

## Related SKILLs

- `add-diffusion-process` — the `ProcessExt` / `sampler()` contract and
  the diffusion baseline jumps layer onto. Read it first.
- `adding-distribution` — when the jump-size law doesn't exist yet
  (Kou's asymmetric double-exponential is the standing gap).
- `integration-test-writing` — pinned seeds and the all-processes guard.
- `python-bindings` — note that `PyMerton` is **hand-written**, not
  macro-generated, because its jump distribution crosses the PyO3
  boundary as a `CallableDist`.
