//! Euler recursion shared by concrete family and jump/gamma entrypoints.

use metal_oxide_device::{ReadBuffer, WriteBuffer, uninit_array};

use super::args::EulerArgs;
use super::draws;
use super::families::{Family, Inputs};
use super::jumps;
use super::{memory, program};

fn initial_inputs(dt: f32) -> Inputs {
  Inputs {
    dt,
    curve: [0.0; 8],
    count: 0.0,
    jump: 0.0,
    gamma: [0.0; 2],
    uniform: [0.0; 2],
    lift: 0.0,
    convolution: 0.0,
    series: 0.0,
    arrival: 0.0,
    exponential: 0.0,
    series_uniform: [0.0; 2],
    inverse: 0.0,
    spacing: 0.0,
    program: [0.0; 2],
    noise: [0.0; 4],
  }
}

/// # Safety
/// An active curve (`slot < count`) has `steps` values and `step` is in its grid.
unsafe fn curve_value(
  buffer: ReadBuffer<f32>,
  count: u32,
  steps: u32,
  step: u32,
  slot: u32,
) -> f32 {
  if slot < count {
    // SAFETY: the caller guarantees the active curve covers this step.
    unsafe { buffer.load_unchecked(slot * steps + step) }
  } else {
    0.0
  }
}

/// Constant slots keep the curves out of private memory.
///
/// # Safety
/// Curves contain `n_curves * steps` values with at most eight curves.
unsafe fn curves(buffer: ReadBuffer<f32>, count: u32, steps: u32, step: u32) -> [f32; 8] {
  // SAFETY: forwarded from the caller for every slot.
  unsafe {
    [
      curve_value(buffer, count, steps, step, 0),
      curve_value(buffer, count, steps, step, 1),
      curve_value(buffer, count, steps, step, 2),
      curve_value(buffer, count, steps, step, 3),
      curve_value(buffer, count, steps, step, 4),
      curve_value(buffer, count, steps, step, 5),
      curve_value(buffer, count, steps, step, 6),
      curve_value(buffer, count, steps, step, 7),
    ]
  }
}

/// Constant components keep the reported values out of private memory.
///
/// # Safety
/// Output covers `components * paths * steps` values. The family has at most
/// four reported components, and path and step select a valid output cell.
unsafe fn write(
  output: WriteBuffer<f32>,
  paths: u32,
  steps: u32,
  path: u32,
  step: u32,
  components: u32,
  values: [f32; 4],
) {
  let plane = paths * steps;
  let cell = path * steps + step;
  // SAFETY: every thread owns all components and grid points of its path.
  unsafe {
    if components > 0 {
      output.store_unchecked(cell, values[0]);
    }
    if components > 1 {
      output.store_unchecked(plane + cell, values[1]);
    }
    if components > 2 {
      output.store_unchecked(2 * plane + cell, values[2]);
    }
    if components > 3 {
      output.store_unchecked(3 * plane + cell, values[3]);
    }
  }
}

/// # Safety
/// The shape matches the family and jump/gamma flags. Output, parameter,
/// increment and curve buffers cover the launch; their sizes fit u32.
/// `steps` is nonzero, gamma_law is at most two, and increments is at most two;
/// `FRACTIONAL` holds exactly when increments is nonzero.
/// Scratch capacities match the family. Active lift, history, series and
/// table dimensions fit them; programs satisfy the fixed-stack encoding.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run<
  F: Family,
  const JUMPS: bool,
  const GAMMA: bool,
  const LIFT: usize,
  const BLOCK: usize,
  const FRACTIONAL: bool,
>(
  output: WriteBuffer<f32>,
  params: ReadBuffer<f32>,
  args: EulerArgs,
  increments: ReadBuffer<f32>,
  curve: ReadBuffer<f32>,
  lift_decay: ReadBuffer<f32>,
  lift_weight: ReadBuffer<f32>,
  lift_drift_scale: ReadBuffer<f32>,
  program: ReadBuffer<f32>,
) {
  let path = crate::index();
  if path >= args.paths || args.steps == 0 {
    return;
  }
  let global = args.first_path.wrapping_add(path);
  let mut state = args.x0;
  let mut input = initial_inputs(args.dt);
  let lift_nodes = if args.lift_n < LIFT as u32 {
    args.lift_n
  } else {
    LIFT as u32
  };
  // SAFETY: only the active lift nodes are read, and they are zeroed first.
  let mut drift_history: [f32; LIFT] = unsafe { uninit_array() };
  let mut shock_history: [f32; LIFT] = unsafe { uninit_array() };
  if F::HAS_LIFT && args.has_lift != 0 {
    for node in 0u32..lift_nodes {
      drift_history[node as usize] = 0.0;
      shock_history[node as usize] = 0.0;
    }
  }
  // SAFETY: series, table and history clauses write each cell before reading it.
  let mut block: [f32; BLOCK] = unsafe { uninit_array() };
  let mut table_position = 1u32;
  // SAFETY: launch validation establishes every buffer's extent.
  unsafe {
    input.curve = curves(curve, args.n_curves, args.steps, 0);
    if F::HAS_SERIES && args.series_n != 0 {
      if args.series_live == 0 {
        let cells = if args.steps < BLOCK as u32 {
          args.steps
        } else {
          BLOCK as u32
        };
        for cell in 0u32..cells {
          block[cell as usize] = 0.0;
        }
      }
      let mut arrival = 0.0;
      let mut j = 1u32;
      while j <= args.series_n {
        let key = memory::key(global, j);
        arrival += memory::gap(key, args.seed);
        if args.series_live != 0 {
          if j >= 1 && j <= BLOCK as u32 {
            block[(j - 1) as usize] = arrival;
          }
        } else {
          let term = memory::term(input, key, args.seed, arrival);
          let cell = memory::cell(key, args.seed, args.steps);
          if cell < BLOCK as u32 {
            block[cell as usize] += F::series(params, args.x0, term)[0];
          }
        }
        j += 1;
      }
    }
    if F::HAS_TABLE && args.table_n >= 2 && args.table_n <= BLOCK as u32 {
      let horizon = args.dt * (args.steps - 1) as f32;
      let mut maximum = memory::table_extent(args, horizon);
      for attempt in 0u32..10 {
        input.spacing = maximum / (args.table_n - 1) as f32;
        block[0] = 0.0;
        for node in 1u32..args.table_n {
          let mut draw = input;
          draw.series_uniform = memory::table_uniforms(global, attempt, node, args.seed);
          block[node as usize] = block[(node - 1) as usize] + F::table(params, args.x0, draw)[0];
        }
        if block[(args.table_n - 1) as usize] >= horizon {
          break;
        }
        if attempt < 9 {
          maximum *= 2.0;
        }
      }
    }
    let reported = F::report(params, state, input);
    if args.step_first == 0 {
      write(
        output,
        args.paths,
        args.steps,
        path,
        0,
        F::COMPONENTS,
        reported,
      );
    }
    for step in (if args.step_first == 0 { 1 } else { 0 })..args.steps {
      let key = draws::cell(global, step);
      // Fractional increments replace the first `increments` hashed draws.
      if F::NOISES > 0 && !FRACTIONAL {
        input.noise[0] = args.sqrt_dt * draws::noise(key, 0, args.seed);
      }
      if F::NOISES > 1 && (!FRACTIONAL || args.increments < 2) {
        input.noise[1] = args.sqrt_dt * draws::noise(key, 1, args.seed);
      }
      if F::NOISES > 2 {
        input.noise[2] = args.sqrt_dt * draws::noise(key, 2, args.seed);
      }
      if F::NOISES > 3 {
        input.noise[3] = args.sqrt_dt * draws::noise(key, 3, args.seed);
      }
      if FRACTIONAL {
        let length = if args.step_first == 0 {
          args.steps - 1
        } else {
          args.steps
        };
        let index = if args.step_first == 0 { step - 1 } else { step };
        input.noise[0] = increments.load_unchecked(path * args.increments * length + index);
        if args.increments > 1 {
          input.noise[1] = increments.load_unchecked((path * args.increments + 1) * length + index);
        }
      }
      input.curve = curves(curve, args.n_curves, args.steps, step);
      if F::HAS_UNIFORMS {
        input.uniform = [
          draws::uniform(key, 2_135_587_861, args.seed, 2_654_435_761),
          draws::uniform(key, 3_266_489_917, args.seed, 2_654_435_761),
        ];
      }
      if F::HAS_SERIES && args.series_n != 0 {
        input.series = 0.0;
        if args.series_live != 0 {
          let terms = if args.series_n < BLOCK as u32 {
            args.series_n
          } else {
            BLOCK as u32
          };
          let mut j = 1u32;
          while j <= terms {
            let series_key = memory::key(global, j);
            if memory::cell(series_key, args.seed, args.steps) == step {
              input = memory::term(input, series_key, args.seed, block[(j - 1) as usize]);
              input.series += F::series(params, state, input)[0];
            }
            j += 1;
          }
          input.arrival = 0.0;
          input.exponential = 0.0;
          input.series_uniform = [0.0; 2];
        } else if step < BLOCK as u32 {
          input.series = block[step as usize];
        }
      }
      if F::HAS_TABLE && args.table_n >= 2 && args.table_n <= BLOCK as u32 {
        let time = args.dt * step as f32;
        while table_position < args.table_n {
          let before = block[table_position as usize] < time;
          if !before {
            break;
          }
          table_position += 1;
        }
        if table_position >= args.table_n {
          input.inverse = input.spacing * (args.table_n - 1) as f32;
        } else if table_position >= 1 {
          let previous = block[(table_position - 1) as usize];
          let current = block[table_position as usize];
          input.inverse = if current <= previous {
            input.spacing * table_position as f32
          } else {
            input.spacing * (table_position - 1) as f32
              + (time - previous) / (current - previous) * input.spacing
          };
        }
      }
      if JUMPS && args.has_jumps != 0 && (F::READS_COUNT || F::READS_JUMP || F::READS_GAMMA) {
        input.count = draws::poisson(key, args.seed, args.jump_lambda, args.dt);
      }
      if GAMMA && F::READS_GAMMA {
        if args.gamma_law > 0 {
          let shape = args.g1_shape + args.g1_per * input.count;
          input.gamma[0] = draws::gamma(key, args.seed, 0, shape, args.g1_scale);
        }
        if args.gamma_law > 1 {
          let shape = args.g2_shape + args.g2_per * input.count;
          input.gamma[1] = draws::gamma(key, args.seed, 1, shape, args.g2_scale);
        }
      }
      if JUMPS && F::READS_JUMP {
        input.jump = jumps::draw(
          key,
          args.seed,
          args.jump_law,
          args.jump_a,
          args.jump_b,
          args.jump_c,
          input.count,
        );
      }
      if F::HAS_PROGRAM {
        input.program = program::evaluate(program, args.program_n, input.curve[0], state[0]);
      }
      let mut lift = [0.0; 4];
      if F::HAS_LIFT && args.has_lift != 0 {
        lift = F::lift(params, state, input);
        let mut history = 0.0;
        for node in 0u32..lift_nodes {
          history += lift_weight.load_unchecked(node)
            * (drift_history[node as usize] + shock_history[node as usize]);
        }
        input.lift =
          args.lift_x0 + args.lift_db * lift[0] + history + args.lift_fb * lift[1] * lift[2];
      }
      if F::HAS_HISTORY && args.hist_slot != u32::MAX {
        let previous = if args.step_first != 0 { step } else { step - 1 };
        let pushed = F::history(params, state, input)[0];
        if previous < BLOCK as u32 {
          block[previous as usize] = pushed;
          let weights = args.hist_slot * args.steps;
          let mut convolution = 0.0;
          let mut position = previous;
          loop {
            convolution +=
              curve.load_unchecked(weights + (previous - position)) * block[position as usize];
            if position == 0 {
              break;
            }
            position -= 1;
          }
          input.convolution = convolution;
        }
      }
      state = F::step(params, state, input);
      if F::HAS_LIFT && args.has_lift != 0 {
        for node in 0u32..lift_nodes {
          drift_history[node as usize] = lift_decay.load_unchecked(node)
            * drift_history[node as usize]
            + lift_drift_scale.load_unchecked(node) * lift[0];
          shock_history[node as usize] =
            lift_decay.load_unchecked(node) * (shock_history[node as usize] + lift[1] * lift[2]);
        }
      }
      let reported = F::report(params, state, input);
      write(
        output,
        args.paths,
        args.steps,
        path,
        step,
        F::COMPONENTS,
        reported,
      );
    }
  }
}
