---
name: add-jump-process
description: How to add a jump-diffusion / Lévy / compound-Poisson process to stochastic-rs-stochastic. Invoke for Merton-jump, Kou-jump, Bates-style models, or for layering jumps onto an existing diffusion (GBM → MJD, Heston → Bates).
---

# Add jump process — stochastic-rs-stochastic

A jump process in `stochastic-rs-stochastic` is parameterised by a
**generic** jump-size distribution `D: Distribution<T> + Send + Sync`,
keeping the jump kernel orthogonal from the diffusion. Compound-Poisson
arrivals are handled by `crate::process::compound_poisson::CompoundPoisson`,
which the new process composes.

The §5.5 trap (rc.0 17-panic class) shipped because jump-driver
constructors silently accepted invalid parameter combinations
(`r > 0`, `r_f > r`, `b ≠ r - r_f`, `mu` out of distribution support).
This SKILL codifies the generic-D pattern, the
characteristic-function consistency check, and the
*construction-time* validation that prevents that class of failure.

## 1. The pattern: composition of `CompoundPoisson<D>`

```rust
// stochastic-rs-stochastic/src/jump/mjd.rs (Merton jump-diffusion)

use crate::process::compound_poisson::CompoundPoisson;
use stochastic_rs_distributions::SimdNormal;

pub struct MertonJumpDiffusion<T: FloatExt, S: SeedExt = Unseeded> {
    pub mu: T, pub sigma: T,        // diffusion parameters
    pub n: usize, pub x0: Option<T>, pub t: Option<T>,
    pub seed: S,
    /// Compound-Poisson jump component, parameterised by the jump-size
    /// distribution (for Merton: lognormal; for Kou: double-exponential).
    jumps: CompoundPoisson<T, SimdNormal<T>>,
}

impl<T: FloatExt> MertonJumpDiffusion<T> {
    pub fn new(
        mu: T, sigma: T, lambda: T,         // diffusion + jump intensity
        jump_mu: T, jump_sigma: T,           // jump-size lognormal params
        n: usize, x0: Option<T>, t: Option<T>,
    ) -> Self {
        // Validation at construction — see §3
        assert!(sigma > T::zero(), "sigma must be > 0");
        assert!(lambda >= T::zero(), "jump intensity λ must be >= 0");
        assert!(jump_sigma > T::zero(), "jump-size sigma must be > 0");

        let jump_dist = SimdNormal::new(
            jump_mu.to_f64().unwrap(),
            jump_sigma.to_f64().unwrap(),
        );
        Self {
            mu, sigma, n, x0, t,
            seed: Unseeded,
            jumps: CompoundPoisson::new(lambda, jump_dist, n - 1, t),
        }
    }
}
```

The `CompoundPoisson` driver provides:
- `sample_increments(seed) -> Vec<T>`: per-step jump sums (zero where
  no jump occurred in that step).
- `sample_arrival_times(seed) -> Vec<T>`: exact jump times
  (Poisson-driven), useful for debugging.

For multi-asset jumps with cross-correlated jump sizes (e.g. Bates
with correlated price/vol jumps), use `CompoundPoisson<T, MultivariateD>`
where `MultivariateD: Distribution<[T; K]>`.

## 2. The sample step

```rust
impl<T: FloatExt, S: SeedExt> ProcessExt<T> for MertonJumpDiffusion<T, S> {
    type Output = Array1<T>;
    fn sample(&self) -> Self::Output {
        let t = self.t.unwrap_or(T::one());
        let dt = t / T::from_usize_(self.n - 1);
        let seed = self.seed.derive();

        // Two RNG streams: diffusion noise + jump increments. Derive
        // child seeds so they're independent.
        let diffusion_seed = seed.advance(0xD1FF_0000);
        let jump_seed = seed.advance(0x1ABE_0000);

        let mut path = Array1::<T>::zeros(self.n);
        path[0] = self.x0.unwrap_or(T::zero());

        let jumps = self.jumps.sample_increments(&jump_seed);
        let mut diff_rng = diffusion_seed.into_rng();

        for i in 1..self.n {
            let z = StandardNormal.sample(&mut diff_rng);
            let z = T::from_f64_fast(z);

            // Lévy-Khintchine compensator for risk-neutral drift:
            // E[exp(jump) - 1] = exp(jump_mu + 0.5 * jump_sigma^2) - 1
            let kappa_bar = (self.jump_mu + T::from_f64_fast(0.5) * self.jump_sigma.powi(2)).exp()
                - T::one();
            let compensator = self.lambda * kappa_bar;

            path[i] = path[i-1]
                + (self.mu - compensator) * dt
                + self.sigma * dt.sqrt() * z
                + jumps[i-1];
        }
        path
    }
}
```

Key: the **compensator** subtraction. Risk-neutral pricing requires
`E[exp(X_t)]` to grow at rate `r - q`, so the deterministic drift must
absorb the expected jump contribution. Forgetting this is the most
common silent-correctness bug in jump-process implementations.

## 3. Construction-time parameter validation (mandatory)

The §5.5 trap was 17 panic-classes that all stemmed from invalid
parameters slipping past construction:

```rust
pub fn new(
    r: T,            // domestic / risk-free
    r_f: T,          // foreign (FX) or dividend
    b: T,            // cost-of-carry; should equal r - r_f
    mu: T,           // jump-size mean
    // ...
) -> Self {
    // Mandatory at construction:
    assert!(r.is_finite() && r >= T::zero(), "r must be finite and >= 0");
    assert!(r_f.is_finite(), "r_f must be finite");
    assert!(
        (b - (r - r_f)).abs() < T::from_f64_fast(1e-9),
        "cost-of-carry b={b} must equal r - r_f = {} (within 1e-9)",
        r - r_f
    );
    assert!(mu.is_finite(), "jump mean mu must be finite");
    // ...
}
```

The class of bug being prevented: a calibrator emits `(r=0.05, r_f=0.02,
b=0.0)` (forgetting `b = r - r_f`); the pricer silently accepts and
mis-prices everything by ~0.03 per year. Catching at construction
forces the calibrator to expose the bug as a panic in tests.

## 4. Characteristic-function consistency

Every jump process should expose `characteristic_function(u, t)` if
the corresponding pricer (Carr-Madan, Lewis, Cosine) needs it. The
Lévy-Khintchine triplet (`gamma`, `sigma`, `nu`) **must** be consistent
with the SDE drift / diffusion / jump kernel:

- `gamma_drift = mu - lambda * E[exp(J) - 1]` (the compensated drift).
- `sigma_diff = sigma` (the Brownian variance is unchanged by jumps).
- `nu(dx) = lambda * f_J(x) dx` (the Lévy measure).

Test: instantiate the process, simulate `M = 50_000` paths, and compare
empirical `E[exp(i u X_T)]` to the Lévy-Khintchine ChF on a strike
grid. The §5.5 audit found multiple processes whose ChF was internally
consistent but disagreed with the SDE drift.

## 5. CompoundPoisson generic parameter — the contract

`CompoundPoisson<T, D: Distribution<T> + Send + Sync>` takes:

- `T: FloatExt` — the base float type (f32 or f64).
- `D: Distribution<T> + Send + Sync` — the jump-size distribution. Must
  be `Send + Sync` so the parallel sampler (`sample_par`) can broadcast
  across threads.

The `D` parameter is **always** a concrete `SimdXxx<T>` from
`stochastic-rs-distributions`, not a `&dyn Distribution<T>`. Per
`dev-rules`, no `dyn` dispatch where concrete types work — and here
the per-step jump-sample call goes through 4 inner-loop function calls;
inlining the concrete sampler matters.

Three reference distributions:
- `SimdNormal<T>` — Merton (1976) jump-diffusion.
- `SimdDoubleExponential<T>` — Kou (2002) jump-diffusion.
- `SimdNig<T>` — Normal-Inverse-Gaussian (subordinator-style).

Adding a new distribution: see `adding-distribution` SKILL.

## 6. Testing requirements

```rust
#[cfg(test)]
mod tests {
    /// 1. Zero jump intensity → matches the underlying diffusion.
    #[test]
    fn lambda_zero_reduces_to_diffusion() { ... }

    /// 2. Mean over many paths matches Lévy-Khintchine compensator.
    #[test]
    fn mean_matches_compensated_drift() { ... }

    /// 3. ChF empirical vs analytical agreement.
    #[test]
    fn chf_matches_levy_khintchine() { ... }

    /// 4. Construction-time validation rejects b != r - r_f.
    #[test]
    #[should_panic(expected = "cost-of-carry b")]
    fn rejects_inconsistent_carry() { ... }

    /// 5. Seeded determinism.
    #[test]
    fn seeded_is_deterministic() { ... }
}
```

## 7. Anti-patterns

- **Do not** use `Box<dyn Distribution<T>>` for the jump sampler. Use
  the generic `D: Distribution<T> + Send + Sync` parameter — see
  `dev-rules` §3 (no boxed traits when concrete types compose).
- **Do not** forget the Lévy-Khintchine compensator. Risk-neutral
  drift must net out the expected jump.
- **Do not** validate parameters at the call site. All input
  validation belongs in `new(...)`, not in `sample()`.
- **Do not** share an RNG between the diffusion and the jump streams.
  Derive independent child seeds; otherwise variance reduction breaks
  for correlated MC estimators.

## 8. Reference impls

- `MertonJumpDiffusion` (`jump/mjd.rs`) — single-asset GBM + lognormal
  jumps. The reference for path (1).
- `KouJumpDiffusion` (`jump/kou.rs`) — double-exponential jumps; same
  shape as MJD with a different distribution.
- `Bates` (`volatility/bates_svj.rs`) — Heston + lognormal jumps;
  multi-asset extension via `CompoundPoisson<T, BivariateD>`.
- `Fbates` (`volatility/fbates_svj.rs`) — fractional Bates (rough
  Heston + jumps); composes path (1) of `add-fractional-process`.

## Related SKILLs

- `add-diffusion-process` — for the diffusion baseline that jumps layer
  on top of.
- `adding-distribution` — when the jump-size distribution doesn't yet
  exist in `stochastic-rs-distributions`.
- `python-bindings` — `py_process_*!` macro works the same.
