use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::uniform::SimdUniform;

use super::*;
use crate::traits::PathSampler;

/// The host stream for a [`SystemProbe`]: independent normals per noise
/// component through the family's own generated host step.
pub(crate) struct SystemProbeSampler<const D: usize> {
  spec: EulerSpec<f32>,
  x0: [f32; D],
  dt: f32,
  normal: Seeded<SimdNormal<f32>>,
  lift: Option<ProbeLift>,
  program: Option<ProbePrograms>,
}

impl<const D: usize> crate::traits::Sealed for SystemProbeSampler<D> {}

impl<const D: usize> PathSampler<f32> for SystemProbeSampler<D> {
  type Output = [Array1<f32>; D];

  fn sample_into(&mut self, out: &mut [Array1<f32>; D]) {
    let (code, params) = self.spec.encode();
    let family = super::families::Family::from_code(code).expect("a declared family");
    let noises = family.noises();
    let curves = probe_curves();
    let mut state = [0.0f32; 4];
    state[..D].copy_from_slice(&self.x0);
    let mut reported = [0.0f32; 4];
    super::families::host_report(
      family,
      &state,
      &params,
      curves[0][0],
      curves[1][0],
      curves[2][0],
      curves[3][0],
      curves[4][0],
      curves[5][0],
      curves[6][0],
      curves[7][0],
      0.0,
      0.0,
      0.7,
      1.3,
      0.5,
      0.5,
      0.5,
      0.5,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      &mut reported,
    );
    for (c, path) in out.iter_mut().enumerate() {
      path[0] = reported[c];
    }
    let mut draw = vec![0.0f32; noises];
    // The lift's two state vectors, advanced per step exactly as the frame
    // advances them, from the family's own drift, diffusion and shock.
    let nodes = self.lift.as_ref().map_or(0, |l| l.decay.len());
    let (mut lh, mut lj) = (vec![0.0f32; nodes], vec![0.0f32; nodes]);
    // The history a `history` family pushes, convolved with its weight curve
    // exactly as the frame does.
    let hist_slot = family.history_slot();
    let mut past = Vec::<f32>::new();
    // The series a `series` family sums into its grid cells, from its own uniform stream as in
    // the frame's preamble; a live family keeps the terms and sizes them in their cell's step.
    let mut series = vec![0.0f32; N];
    let mut terms: Vec<(f32, f32, f32, f32, usize)> = Vec::new();
    if family.has_series() {
      let mut gj = 0.0f32;
      for term in 1..=PROBE_TERMS {
        let (gap, ej, uj, uv, time) = series_draws(term);
        gj += gap;
        let ratio = time * (N - 1) as f32;
        let cell = (ratio.ceil() as usize).clamp(1, N - 1);
        if family.series_live() {
          terms.push((gj, ej, uj, uv, cell));
        } else {
          let size = super::families::host_series(family, &state, &params, self.dt, gj, ej, uj, uv)
            .expect("a family with a series clause sizes its terms");
          series[cell] += size;
        }
      }
    }
    // The table a `table` family inverts, built exactly as the frame's
    // preamble builds it, and the pointer its steps scan with.
    let horizon = self.dt * (N - 1) as f32;
    let mut table = Vec::<f32>::new();
    let mut spacing = 0.0f32;
    if family.has_table() {
      let mut uniform = SimdUniform::<f32>::new(0.0, 1.0).seeded(&Deterministic::new(13));
      let mut umax = 1.0f32;
      for attempt in 0..10 {
        spacing = umax / (PROBE_TABLE - 1) as f32;
        table = vec![0.0f32; PROBE_TABLE as usize];
        for k in 1..PROBE_TABLE as usize {
          let (uj, uv) = (uniform.sample(), uniform.sample());
          let inc = super::families::host_table(family, &params, self.dt, uj, uv, spacing)
            .expect("a family with a table clause sizes its increments");
          table[k] = table[k - 1] + inc;
        }
        if table[PROBE_TABLE as usize - 1] >= horizon {
          break;
        }
        if attempt < 9 {
          umax *= 2.0;
        }
      }
    }
    let mut pointer = 1usize;
    let mut inverse_at = |ti: f32| -> f32 {
      if table.is_empty() {
        return 0.0;
      }
      while pointer < table.len() && table[pointer] < ti {
        pointer += 1;
      }
      if pointer >= table.len() {
        spacing * (table.len() - 1) as f32
      } else if table[pointer] <= table[pointer - 1] {
        spacing * pointer as f32
      } else {
        spacing * (pointer - 1) as f32
          + (ti - table[pointer - 1]) / (table[pointer] - table[pointer - 1]) * spacing
      }
    };
    for i in 1..N {
      let mut noise = [0.0f32; 4];
      self.normal.fill_slice(&mut draw);
      noise[..noises].copy_from_slice(&draw);
      let (pv, pv2) = self
        .program
        .as_ref()
        .map_or((0.0, 0.0), |p| p.values(curves[0][i], state[0]));
      let (mut lv, mut coefficients) = (0.0f32, [0.0f32; 3]);
      if let Some(lift) = &self.lift {
        coefficients =
          super::families::host_lift(family, &state, &params, self.dt, pv, pv2, &noise);
        let [lf, lg, lsh] = coefficients;
        let hist: f32 = (0..nodes).map(|l| lift.weight[l] * (lh[l] + lj[l])).sum();
        lv = lift.x0 + lift.db * lf + hist + lift.fb * lg * lsh;
      }
      let mut cv = 0.0f32;
      if let Some(slot) = hist_slot {
        let pushed = super::families::host_history(
          family, &state, &params, self.dt, &noise, 0.5, 0.5, 0.5, 0.5,
        )
        .expect("a family with a history clause pushes a value");
        past.push(pushed);
        let weights = &curves[slot as usize];
        let hi = past.len() - 1;
        cv = (0..=hi).map(|k| weights[k] * past[hi - k]).sum();
      }
      let mut next = [0.0f32; 4];
      super::families::host_step(
        family,
        &state,
        &params,
        self.dt,
        curves[0][i],
        curves[1][i],
        curves[2][i],
        curves[3][i],
        curves[4][i],
        curves[5][i],
        curves[6][i],
        curves[7][i],
        0.0,
        0.0,
        0.7,
        1.3,
        0.5,
        0.5,
        0.5,
        0.5,
        lv,
        cv,
        cell_sum(family, &state, &params, self.dt, &terms, &series, i),
        0.0,
        0.0,
        0.0,
        0.0,
        inverse_at(self.dt * i as f32),
        spacing,
        pv,
        pv2,
        &noise,
        &mut next,
      );
      state = next;
      if let Some(lift) = &self.lift {
        let [lf, lg, lsh] = coefficients;
        for l in 0..nodes {
          lh[l] = lift.decay[l] * lh[l] + lift.drift_scale[l] * lf;
          lj[l] = lift.decay[l] * (lj[l] + lg * lsh);
        }
      }
      super::families::host_report(
        family,
        &state,
        &params,
        curves[0][i],
        curves[1][i],
        curves[2][i],
        curves[3][i],
        curves[4][i],
        curves[5][i],
        curves[6][i],
        curves[7][i],
        0.0,
        0.0,
        0.7,
        1.3,
        0.5,
        0.5,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        &mut reported,
      );
      for (c, path) in out.iter_mut().enumerate() {
        path[i] = reported[c];
      }
    }
  }

  fn sample(&mut self) -> [Array1<f32>; D] {
    let mut out = std::array::from_fn(|_| Array1::zeros(N));
    self.sample_into(&mut out);
    out
  }
}

impl<const D: usize> crate::traits::Sealed for SystemProbe<D> {}

impl<const D: usize> ProcessExt<f32> for SystemProbe<D> {
  type Output = [Array1<f32>; D];
  type Sampler<'s>
    = SystemProbeSampler<D>
  where
    Self: 's;

  fn sampler(&self) -> SystemProbeSampler<D> {
    let dt = 1.0 / (N - 1) as f32;
    SystemProbeSampler {
      spec: self.spec,
      x0: self.x0,
      dt,
      normal: SimdNormal::<f32>::new(0.0, dt.sqrt()).seeded(&Deterministic::new(7)),
      lift: self.lift.clone(),
      program: self.program.clone(),
    }
  }
}

impl<const D: usize> EulerSystem<f32, D> for SystemProbe<D> {
  fn euler_spec(&self) -> EulerSpec<f32> {
    self.spec
  }

  fn initial_state(&self) -> [f32; 4] {
    let mut slots = [0.0; 4];
    slots[..D].copy_from_slice(&self.x0);
    slots
  }

  fn grid_points(&self) -> usize {
    N
  }

  fn horizon(&self) -> f32 {
    1.0
  }

  fn device_seed(&self) -> u64 {
    11
  }

  fn curves(&self) -> Option<Vec<Vec<f32>>> {
    Some(probe_curves())
  }

  fn lift_spec(&self) -> Option<crate::euler::LiftSpec<'_, f32>> {
    self.lift.as_ref().map(|l| crate::euler::LiftSpec {
      decay: &l.decay,
      weight: &l.weight,
      drift_scale: &l.drift_scale,
      drift_boundary: l.db,
      diffusion_boundary: l.fb,
      x0: l.x0,
    })
  }

  /// A series family draws the probes' fixed handful of terms per path; the
  /// rest draw none.
  fn series_terms(&self) -> Option<u32> {
    let (code, _) = self.spec.encode();
    super::families::Family::from_code(code)
      .expect("a declared family")
      .has_series()
      .then_some(PROBE_TERMS)
  }

  /// A table family builds the probes' fixed table over the unit extent.
  fn table_spec(&self) -> Option<crate::euler::TableSpec<f32>> {
    let (code, _) = self.spec.encode();
    super::families::Family::from_code(code)
      .expect("a declared family")
      .has_table()
      .then_some(crate::euler::TableSpec {
        points: PROBE_TABLE,
        u_max: 1.0,
      })
  }

  fn program_spec(&self) -> Option<crate::euler::ProgramSpec<'_>> {
    self.program.as_ref().map(ProbePrograms::spec)
  }

  fn jump_intensity(&self) -> Option<f32> {
    Some(PROBE_INTENSITY)
  }

  fn jump_sizes(&self) -> Option<JumpSizes<f32>> {
    Some(probe_sizes(&self.spec))
  }

  fn gamma_draws(&self) -> Option<GammaDraws<f32>> {
    Some(PROBE_GAMMAS)
  }

  fn host_sample(&self) -> [Array1<f32>; D] {
    <Self as ProcessExt<f32>>::sampler(self).sample()
  }
}
