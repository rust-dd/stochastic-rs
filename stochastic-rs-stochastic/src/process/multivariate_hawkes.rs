//! # Multivariate Hawkes Process
//!
//! $$
//! \lambda_i(t) = \mu_i + \sum_{j=1}^{D}\sum_{T_k^j < t} \alpha_{ij}\,e^{-\beta_{ij}(t - T_k^j)},
//! \quad i = 1,\dots,D
//! $$
//!
//! D-dimensional self-exciting point process with exponential kernels and
//! cross-excitation. Stationarity requires $\rho(\Gamma) < 1$ where
//! $\Gamma_{ij} = \alpha_{ij}/\beta_{ij}$.
//!
//! Simulated via multivariate Ogata thinning.
//!
//! Reference:
//! - Hawkes (1971), "Spectra of some self-exciting and mutually exciting point processes"
//! - Bacry, Mastromatteo, Muzy (2015), "Hawkes processes in finance", arXiv:1502.04592

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::device::DeviceError;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// D-dimensional Hawkes process with exponential kernels.
///
/// Output: `ProcessExt::Output = Vec<Array1<T>>` — one `Array1<T>` of
/// event times per component (length D), each component's vector sized to
/// however many events it actually had (event counts differ per
/// component, so this is not a fixed-shape `Array2<T>`).
pub struct MultivariateHawkes<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Baseline intensities $\mu_i > 0$, length D.
  pub mu: Array1<T>,
  /// Excitation matrix $\alpha_{ij} \ge 0$, shape (D, D): size of the
  /// intensity jump on component `i` triggered by an event on component
  /// `j`.
  pub alpha: Array2<T>,
  /// Decay matrix $\beta_{ij} > 0$, shape (D, D): rate at which the
  /// `j`-to-`i` excitation from `alpha` fades back toward `mu_i`.
  pub beta: Array2<T>,
  /// Time horizon — event generation stops once simulated time exceeds
  /// this value. Ignored in count mode.
  pub t_max: T,
  /// Optional fixed number of events after the origins, across all
  /// components: count mode, the one a device can run. `None` samples over
  /// the horizon.
  pub n: Option<usize>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> MultivariateHawkes<T, S> {
  pub fn new(mu: Array1<T>, alpha: Array2<T>, beta: Array2<T>, t_max: T, seed: S) -> Self {
    let d = mu.len();
    assert_eq!(alpha.shape(), [d, d], "alpha must be (D, D)");
    assert_eq!(beta.shape(), [d, d], "beta must be (D, D)");
    Self {
      backend: Cpu,
      mu,
      alpha,
      beta,
      t_max,
      n: None,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> MultivariateHawkes<T, S, B> {
  /// Count mode: exactly `n` events after the origins, across all
  /// components, the horizon ignored.
  pub fn with_count(mut self, n: usize) -> Self {
    self.n = Some(n);
    self
  }

  /// The number of components, `D`.
  pub fn dim(&self) -> usize {
    self.mu.len()
  }
}

backend_switch!([T: FloatExt, S: SeedExt] MultivariateHawkes<T, S> { mu, alpha, beta, t_max, n, seed } via euler);

/// The device's two rows — every event's time and its component — back into
/// one event list per component, each opening at the origin as the host's do.
fn rows_to_components<T: FloatExt>(rows: [Array1<T>; 2], d: usize) -> Vec<Array1<T>> {
  let [times, marks] = rows;
  let mut events: Vec<Vec<T>> = (0..d).map(|_| vec![T::zero()]).collect();
  for j in 1..times.len() {
    let k = marks[j].to_f64().unwrap_or(0.0).round().max(0.0) as usize;
    events[k.min(d - 1)].push(times[j]);
  }
  events.into_iter().map(Array1::from_vec).collect()
}

/// The Euler engine's view of the process: the events as two rows, time and
/// mark, one event per grid step.
#[doc(hidden)]
pub struct MultivariateHawkesLaunch<'a, T: FloatExt, S: SeedExt, B>(
  &'a MultivariateHawkes<T, S, B>,
);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for MultivariateHawkesLaunch<'_, T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = MultivariateHawkesLaunchSampler<'s, T, S>
  where
    Self: 's;

  fn sampler(&self) -> MultivariateHawkesLaunchSampler<'_, T, S> {
    MultivariateHawkesLaunchSampler {
      inner: <MultivariateHawkes<T, S, B> as ProcessExt<T>>::sampler(self.0),
    }
  }
}

#[doc(hidden)]
pub struct MultivariateHawkesLaunchSampler<'a, T: FloatExt, S: SeedExt> {
  inner: MultivariateHawkesSampler<'a, T, S>,
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for MultivariateHawkesLaunchSampler<'_, T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    *out = self.sample();
  }

  /// The host's per-component lists merged into time order, with the mark.
  fn sample(&mut self) -> [Array1<T>; 2] {
    let components = self.inner.sample_inner();
    let mut events: Vec<(T, usize)> = components
      .iter()
      .enumerate()
      .flat_map(|(k, times)| times.iter().skip(1).map(move |&t| (t, k)))
      .collect();
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut times = vec![T::zero()];
    let mut marks = vec![T::zero()];
    for (t, k) in events {
      times.push(t);
      marks.push(T::from_usize_(k));
    }
    [Array1::from_vec(times), Array1::from_vec(marks)]
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for MultivariateHawkesLaunch<'_, T, S, B>
{
  /// The baselines, the excitation matrix and one decay per target, a missing
  /// second component travelling with a zero baseline and zero excitations. A
  /// configuration the family cannot carry never reaches a launch —
  /// [`ProcessExt::sample`] keeps it on the host — so asking here is a caller
  /// bypassing that guard.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let p = self.0;
    assert!(
      p.device_ready(),
      "MultivariateHawkes: a launch carries count mode with at most two components and one decay \
       per target; sample through `ProcessExt`, which keeps the rest on the host"
    );
    let d = p.dim();
    let at = |i: usize, j: usize| {
      if i < d && j < d {
        p.alpha[[i, j]]
      } else {
        T::zero()
      }
    };
    let mu = |i: usize| if i < d { p.mu[i] } else { T::zero() };
    let beta = |i: usize| if i < d { p.beta[[i, 0]] } else { T::one() };
    crate::euler::EulerSpec::HawkesEvents2 {
      mu: [mu(0), mu(1)],
      alpha: [at(0, 0), at(0, 1), at(1, 0), at(1, 1)],
      beta: [beta(0), beta(1)],
    }
  }

  /// The origin, with no excess intensity yet and the mark at zero.
  fn initial_state(&self) -> [T; 4] {
    [T::zero(); 4]
  }

  /// One step is one event, after the origin.
  fn grid_points(&self) -> usize {
    self.0.n.expect(
      "the Euler engine describes MultivariateHawkes's count mode; horizon mode has no grid",
    ) + 1
  }

  /// One step is one event, so the grid has no horizon of its own; the waits
  /// come from the intensities in the step.
  fn horizon(&self) -> T {
    T::one()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.0.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for MultivariateHawkes<T, S, B>
{
  type Output = Vec<Array1<T>>;
  type Sampler<'s>
    = MultivariateHawkesSampler<'s, T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> MultivariateHawkesSampler<'_, T, S> {
    MultivariateHawkesSampler {
      mu: &self.mu,
      alpha: &self.alpha,
      beta: &self.beta,
      t_max: self.t_max,
      n: self.n,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine in count mode with at most two components and
  /// one decay per target, where every step is one event; anything else keeps
  /// the process on the host, chunked exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> Vec<Array1<T>> {
    if self.device_ready() {
      rows_to_components(
        self.backend.system_sample(&MultivariateHawkesLaunch(self)),
        self.dim(),
      )
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Vec<Array1<T>>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      let d = self.dim();
      self
        .backend
        .system_paths_map(&MultivariateHawkesLaunch(self), m, |rows| {
          f(&rows_to_components(rows.clone(), d))
        })
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Vec<Array1<T>>> {
    if self.device_ready() {
      let d = self.dim();
      self
        .backend
        .system_paths(&MultivariateHawkesLaunch(self), m)
        .into_iter()
        .map(|rows| rows_to_components(rows, d))
        .collect()
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Vec<Array1<T>>, DeviceError> {
    if self.device_ready() {
      Ok(rows_to_components(
        self
          .backend
          .try_system_sample(&MultivariateHawkesLaunch(self))?,
        self.dim(),
      ))
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Vec<Array1<T>>>, DeviceError> {
    if self.device_ready() {
      let d = self.dim();
      Ok(
        self
          .backend
          .try_system_paths(&MultivariateHawkesLaunch(self), m)?
          .into_iter()
          .map(|rows| rows_to_components(rows, d))
          .collect(),
      )
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// Whether a device can run this process: count mode — the horizon mode's
  /// length is itself random and has no grid — with at most two components,
  /// one decay per target (`β_ij` constant along each row) and positive
  /// baselines. The family superposes each target's exact excess clock with
  /// the baselines' joint Poisson clock, which is what one decay per target
  /// makes closed-form; per-pair decays keep the process on the host.
  fn device_ready(&self) -> bool {
    let d = self.dim();
    self.n.is_some()
      && d <= 2
      && (0..d).all(|i| (0..d).all(|j| self.beta[[i, j]] == self.beta[[i, 0]]))
      && self.mu.iter().all(|&m| m > T::zero())
  }
}

/// Reusable [`MultivariateHawkes`] sampling state: borrows the baseline
/// intensities, excitation and decay matrices and owns the seed source. The
/// component count is a runtime property of `mu`, and event counts differ per
/// path, so each call rebuilds the output `Vec`; the RNG is rebuilt from the
/// owned seed each call, exactly as the legacy `sample` body did.
#[doc(hidden)]
pub struct MultivariateHawkesSampler<'a, T: FloatExt, S: SeedExt> {
  mu: &'a Array1<T>,
  alpha: &'a Array2<T>,
  beta: &'a Array2<T>,
  t_max: T,
  n: Option<usize>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> MultivariateHawkesSampler<'_, T, S> {
  /// Multivariate Ogata thinning, over the horizon or until the event count.
  fn sample_inner(&mut self) -> Vec<Array1<T>> {
    let mut rng = self.seed.rng();
    let d = self.mu.len();

    // S[i][j] = running self-exciting component from source j to target i
    let mut s = vec![vec![T::zero(); d]; d];
    let mut t = T::zero();
    let mut events: Vec<Vec<T>> = (0..d).map(|_| vec![T::zero()]).collect();
    let mut produced = 0usize;

    loop {
      match self.n {
        Some(n) if produced >= n => break,
        None if t >= self.t_max => break,
        _ => {}
      }
      // Compute component intensities and total upper bound
      let mut lambdas = vec![T::zero(); d];
      for i in 0..d {
        lambdas[i] = self.mu[i];
        for j in 0..d {
          lambdas[i] += s[i][j];
        }
      }
      let lambda_bar: T = lambdas.iter().copied().sum();
      if lambda_bar <= T::zero() {
        break;
      }

      // Propose next event time
      let u = T::one() - T::sample_uniform_simd(&mut rng);
      let dt = -u.ln() / lambda_bar;
      t += dt;

      if self.n.is_none() && t >= self.t_max {
        break;
      }

      // Decay all S components
      for i in 0..d {
        for j in 0..d {
          s[i][j] = s[i][j] * (-self.beta[[i, j]] * dt).exp();
        }
      }

      // Recompute intensities at proposed time
      for i in 0..d {
        lambdas[i] = self.mu[i];
        for j in 0..d {
          lambdas[i] += s[i][j];
        }
      }

      // Accept/reject and assign to component
      let v = T::sample_uniform_simd(&mut rng) * lambda_bar;
      let mut cumsum = T::zero();
      for i in 0..d {
        cumsum += lambdas[i];
        if v <= cumsum {
          // Event accepted on component i
          events[i].push(t);
          produced += 1;
          // Excite all components from source i
          for k in 0..d {
            s[k][i] += self.alpha[[k, i]];
          }
          break;
        }
      }
    }

    events.into_iter().map(Array1::from_vec).collect()
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for MultivariateHawkesSampler<'_, T, S> {
  type Output = Vec<Array1<T>>;

  fn sample_into(&mut self, out: &mut Vec<Array1<T>>) {
    *out = self.sample_inner();
  }

  fn sample(&mut self) -> Vec<Array1<T>> {
    self.sample_inner()
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array2;
  use ndarray::array;

  use super::*;

  #[test]
  fn bivariate_hawkes_runs() {
    let mu = array![1.0_f64, 1.5];
    let alpha = Array2::from_shape_vec((2, 2), vec![0.3, 0.1, 0.2, 0.4]).unwrap();
    let beta = Array2::from_shape_vec((2, 2), vec![2.0, 2.0, 2.0, 2.0]).unwrap();
    let h = MultivariateHawkes::new(mu, alpha, beta, 10.0, Unseeded);
    let events = h.sample();
    assert_eq!(events.len(), 2);
    assert!(events[0].len() > 1, "component 0 should have events");
    assert!(events[1].len() > 1, "component 1 should have events");
  }

  #[test]
  fn cross_excitation_increases_events() {
    // No cross-excitation
    let mu = array![2.0_f64, 2.0];
    let alpha_diag = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.0, 0.5]).unwrap();
    let beta = Array2::from_shape_vec((2, 2), vec![3.0, 3.0, 3.0, 3.0]).unwrap();
    let h1 = MultivariateHawkes::new(mu.clone(), alpha_diag, beta.clone(), 50.0, Unseeded);

    // With cross-excitation
    let alpha_full = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.5, 0.5]).unwrap();
    let h2 = MultivariateHawkes::new(mu, alpha_full, beta, 50.0, Unseeded);

    let n_trials = 20;
    let avg1: f64 = (0..n_trials)
      .map(|_| (h1.sample()[0].len() + h1.sample()[1].len()) as f64)
      .sum::<f64>()
      / n_trials as f64;
    let avg2: f64 = (0..n_trials)
      .map(|_| (h2.sample()[0].len() + h2.sample()[1].len()) as f64)
      .sum::<f64>()
      / n_trials as f64;

    // Cross-excitation should produce more events on average
    assert!(
      avg2 > avg1 * 0.9,
      "cross-excited avg={avg2:.0} should exceed diagonal avg={avg1:.0}"
    );
  }

  #[test]
  fn univariate_matches_scalar() {
    // D=1 multivariate should behave like the scalar Hawkes
    let mu = array![3.0_f64];
    let alpha = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let beta = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
    let h = MultivariateHawkes::new(mu, alpha, beta, 10.0, Unseeded);
    let events = h.sample();
    assert_eq!(events.len(), 1);
    assert!(events[0].len() > 2, "should have multiple events");
    // Events should be sorted
    for w in events[0].as_slice().unwrap().windows(2) {
      assert!(w[0] <= w[1], "events must be sorted");
    }
  }
}
