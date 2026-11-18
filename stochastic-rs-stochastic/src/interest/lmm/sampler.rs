use ndarray::Axis;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::*;

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Lmm<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Lmm<T, S, B> {
  type Output = Array2<T>;
  type Sampler<'s>
    = LmmSampler<T, S>
  where
    Self: 's;

  /// Derives `self.seed` into the sampler, never clones it, so chunks are decorrelated. It
  /// re-validates `tenor`, `l0` and `sigma` jointly, which the setters leave to sampling.
  fn sampler(&self) -> LmmSampler<T, S> {
    self.validate_lmm_inputs();
    LmmSampler {
      tenor: self.tenor.clone(),
      l0: self.l0.clone(),
      sigma: self.sigma.clone(),
      chol: self.chol.clone(),
      n: self.n,
      t: self.t,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine when the curve fits a launch's four slots; a wider one stays on
  /// the host, chunked as [`ProcessExt`] chunks.
  fn sample(&self) -> Array2<T> {
    if self.device_ready() {
      rows_to_matrix(self.backend.system_sample(&LmmLaunch(self)), self.l0.len())
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array2<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      let rows = self.l0.len();
      self.backend.system_paths_map(&LmmLaunch(self), m, |slots| {
        f(&rows_to_matrix(slots.clone(), rows))
      })
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array2<T>> {
    if self.device_ready() {
      let rows = self.l0.len();
      self
        .backend
        .system_paths(&LmmLaunch(self), m)
        .into_iter()
        .map(|slots| rows_to_matrix(slots, rows))
        .collect()
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array2<T>, crate::device::DeviceError> {
    if self.device_ready() {
      let slots = self.backend.try_system_sample(&LmmLaunch(self))?;
      Ok(rows_to_matrix(slots, self.l0.len()))
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array2<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      let rows = self.l0.len();
      Ok(
        self
          .backend
          .try_system_paths(&LmmLaunch(self), m)?
          .into_iter()
          .map(|slots| rows_to_matrix(slots, rows))
          .collect(),
      )
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: at most as many forwards as a
  /// launch has state slots and shocks. A wider curve samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.l0.len() <= crate::euler::CORRELATED_STREAMS))
      .then_some("more forwards than a launch has state slots and shocks")
  }
}

/// Reusable [`Lmm`] sampling state: the tenor, the initial curve, the volatilities, the Cholesky
/// factor and the seed source; [`LmmSampler::fill_matrix`] holds the drift coupling and the step.
#[doc(hidden)]
#[derive(Debug)]
pub struct LmmSampler<T: FloatExt, S: SeedExt> {
  tenor: Array1<T>,
  l0: Array1<T>,
  sigma: Array1<T>,
  chol: Option<Array2<T>>,
  n: usize,
  t: Option<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> LmmSampler<T, S> {
  fn fill_matrix(&mut self, path: &mut Array2<T>) {
    let m = self.l0.len();
    let n_steps = self.n;
    if n_steps == 0 {
      return;
    }

    for (n, &l0) in self.l0.iter().enumerate() {
      path[(n, 0)] = l0;
    }
    if n_steps == 1 {
      return;
    }

    let t_max = self.tenor[m];
    let horizon = self.t.unwrap_or(t_max);
    assert!(
      horizon > T::zero() && horizon <= t_max,
      "horizon must satisfy `horizon > T::zero() && horizon <= t_max`, got horizon = {horizon:?}, t_max = {t_max:?}"
    );

    let n_increments = n_steps - 1;
    let dt = horizon / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let half = T::from_f64_fast(0.5);

    // Day-count fractions delta_j = T_{j+1} - T_j (length M).
    let delta: Array1<T> = (0..m).map(|j| self.tenor[j + 1] - self.tenor[j]).collect();

    // Standard normal innovations for one full simulation: M factors × n_increments.
    let mut normal = SimdNormal::<T>::new(T::zero(), T::one()).seeded(&self.seed);
    let mut eps = Array2::<T>::zeros((m, n_increments));
    {
      let buf = eps
        .as_slice_mut()
        .expect("LMM eps buffer must be contiguous");
      normal.fill_slice(buf);
    }

    // Pre-compute correlated standard normals: Z_n,k = Σ_j chol_{n,j} eps_{j,k}
    // (only when chol is provided; identity otherwise).
    let z_corr: Array2<T> = match &self.chol {
      Some(chol) => chol.dot(&eps),
      None => eps,
    };

    // Step loop. Each step uses spot-LIBOR drift with the active set
    // η(t) ≤ n, where η(t) = min{n : T_n > t}.
    let mut t_now = T::zero();
    let mut eta: usize = 0;
    let mut prev = self.l0.clone();
    for k in 0..n_increments {
      // Advance η(t) so that T_{eta} > t_now (i.e. Libor 0..eta have reset).
      while eta < m && self.tenor[eta] <= t_now {
        eta += 1;
      }
      let eta_idx = eta.saturating_sub(1).min(m); // index into Libor vector

      // μ_n(t) = σ_n · Σ_{j=eta_idx..=n} (δ_j ρ_{n,j} σ_j L_j) / (1 + δ_j L_j) for each alive n;
      // under the identity correlation the sum is its j = n term.
      let mut next = prev.clone();
      for n in eta_idx..m {
        if self.sigma[n] <= T::zero() {
          // Zero-vol Libor stays at its current value.
          continue;
        }
        // Drift μ_n.
        let mut drift = T::zero();
        match &self.chol {
          Some(_chol) => {
            // The full correlated sum, with ρ_{n,j} = Σ_p chol_{n,p} chol_{j,p} rebuilt here.
            let chol_ref = self.chol.as_ref().unwrap();
            for j in eta_idx..=n {
              let mut rho_nj = T::zero();
              for p in 0..=j.min(n) {
                rho_nj += chol_ref[(n, p)] * chol_ref[(j, p)];
              }
              let denom = T::one() + delta[j] * prev[j];
              drift += (delta[j] * rho_nj * self.sigma[j] * prev[j]) / denom;
            }
            drift = drift * self.sigma[n];
          }
          None => {
            // Identity correlation: only j = n term survives.
            let denom = T::one() + delta[n] * prev[n];
            drift = self.sigma[n] * self.sigma[n] * delta[n] * prev[n] / denom;
          }
        }

        // Log-Euler step with frozen drift.
        let z_nk = z_corr[(n, k)];
        let log_inc =
          (drift - half * self.sigma[n] * self.sigma[n]) * dt + self.sigma[n] * sqrt_dt * z_nk;
        next[n] = prev[n] * log_inc.exp();
      }

      // A Libor whose reset has passed (T_n ≤ t_now + dt) is frozen at its reset value:
      // `prev[n]` is left untouched from then on.
      let t_next = t_now + dt;
      for n in 0..m {
        if self.tenor[n] <= t_now {
          // Already reset — keep prev[n]. (Skip in next pass via eta.)
          next[n] = prev[n];
        }
      }

      for n in 0..m {
        path[(n, k + 1)] = next[n];
      }
      prev = next;
      t_now = t_next;
    }

    let _ = path.axis_iter(Axis(0));
    let _ = path.slice(s![.., ..]);
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for LmmSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for LmmSampler<T, S> {
  type Output = Array2<T>;

  fn sample_into(&mut self, out: &mut Array2<T>) {
    self.fill_matrix(out);
  }

  fn sample(&mut self) -> Array2<T> {
    let mut path = Array2::<T>::zeros((self.l0.len(), self.n));
    self.fill_matrix(&mut path);
    path
  }
}
