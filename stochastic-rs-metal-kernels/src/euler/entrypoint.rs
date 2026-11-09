//! Concrete public names and compile-time scratch capacities.

macro_rules! entrypoint {
  ($name:ident, $family:ident, $jumps:literal, $gamma:literal) => {
    /// # Safety
    /// The buffers cover the launch dimensions and family parameters, and
    /// output is disjoint from every input. The family matches the entrypoint,
    /// as do jump/gamma flags it reads. Lift nodes fit 176 entries; history/series/table
    /// scratch fits 512 entries; programs obey the two-program, 62-opcode,
    /// eight-stack-slot encoding. Counts and buffer indices fit u32.
    #[metal_oxide_device::kernel]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn $name(
      output: metal_oxide_device::WriteBuffer<f32>,
      params: metal_oxide_device::ReadBuffer<f32>,
      args: $crate::euler::EulerArgs,
      increments: metal_oxide_device::ReadBuffer<f32>,
      curve: metal_oxide_device::ReadBuffer<f32>,
      lift_decay: metal_oxide_device::ReadBuffer<f32>,
      lift_weight: metal_oxide_device::ReadBuffer<f32>,
      lift_drift_scale: metal_oxide_device::ReadBuffer<f32>,
      program: metal_oxide_device::ReadBuffer<f32>,
    ) {
      use $crate::euler::families::{Family, $family};
      // SAFETY: the entrypoint contract provides the family frame's buffers.
      unsafe {
        if args.increments == 0 {
          $crate::euler::frame::run::<
            $family,
            $jumps,
            $gamma,
            { $family::LIFT_CAPACITY },
            { $family::BLOCK_CAPACITY },
            false,
          >(
            output,
            params,
            args,
            increments,
            curve,
            lift_decay,
            lift_weight,
            lift_drift_scale,
            program,
          );
        } else {
          $crate::euler::frame::run::<
            $family,
            $jumps,
            $gamma,
            { $family::LIFT_CAPACITY },
            { $family::BLOCK_CAPACITY },
            true,
          >(
            output,
            params,
            args,
            increments,
            curve,
            lift_decay,
            lift_weight,
            lift_drift_scale,
            program,
          );
        }
      }
    }
  };
}
