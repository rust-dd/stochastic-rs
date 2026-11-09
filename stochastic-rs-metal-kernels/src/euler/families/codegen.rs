//! Generate concrete family helpers without duplicating their formulas.

macro_rules! device_families {
  (definitions) => {
    euler_family_definitions!(device_families);
  };

  (
    step_inputs $inputs:tt;
    $(
      $(#[$meta:meta])*
      $code:literal => $name:ident { $($param:ident),* $(,)? }
        state ($($state:ident),+ $(,)?)
        noise ($($noise:ident),+ $(,)?)
        step { $($step:tt)* }
        report { $($report:tt)* }
        $(lift $lift:tt)?
        $(history $history:tt)?
        $(series $series:tt)?
        $(table $table:tt)?
    ),* $(,)?
  ) => {
    $(
      #[allow(dead_code)]
      pub(crate) struct $name;

      impl Family for $name {
        const COMPONENTS: u32 = device_families!(@count $($report)*);
        const NOISES: u32 = [$(stringify!($noise)),*].len() as u32;
        const HAS_LIFT: bool = device_families!(@present $($lift)?);
        const HAS_HISTORY: bool = device_families!(@present $($history)?);
        const HAS_SERIES: bool = device_families!(@present $($series)?);
        const HAS_TABLE: bool = device_families!(@present $($table)?);
        const HAS_PROGRAM: bool = device_families!(@reads "pv", $($step)* $($report)* $($lift)? $($history)? $($series)? $($table)?)
          || device_families!(@reads "pv2", $($step)* $($report)* $($lift)? $($history)? $($series)? $($table)?);
        const HAS_UNIFORMS: bool = device_families!(@reads "u", $($step)* $($report)* $($lift)? $($history)? $($series)? $($table)?)
          || device_families!(@reads "u2", $($step)* $($report)* $($lift)? $($history)? $($series)? $($table)?);
        const READS_COUNT: bool = device_families!(@reads "nj", $($step)* $($report)* $($lift)? $($history)? $($series)? $($table)?);
        const READS_JUMP: bool = device_families!(@reads "js", $($step)* $($report)* $($lift)? $($history)? $($series)? $($table)?);
        const READS_GAMMA: bool = device_families!(@reads "gm", $($step)* $($report)* $($lift)? $($history)? $($series)? $($table)?)
          || device_families!(@reads "gm2", $($step)* $($report)* $($lift)? $($history)? $($series)? $($table)?);

        device_families!(@method step, $inputs, [$($param)*], [$($state)*], [$($noise)*], $($step)*);
        device_families!(@method report, $inputs, [$($param)*], [$($state)*], [$($noise)*], $($report)*);
        device_families!(@lift $inputs, [$($param)*], [$($state)*], [$($noise)*], $($lift)?);
        device_families!(@history $inputs, [$($param)*], [$($state)*], [$($noise)*], $($history)?);
        device_families!(@series $inputs, [$($param)*], [$($state)*], [$($noise)*], $($series)?);
        device_families!(@table $inputs, [$($param)*], [$($state)*], [$($noise)*], $($table)?);
      }
    )*
  };

  (@present) => { false };
  (@present $clause:tt) => { true };
  (@reads $name:literal, $($body:tt)*) => { definitions::reads(stringify!($($body)*), $name) };

  (@lift $inputs:tt, $params:tt, $state:tt, $noise:tt,) => {};
  (@lift $inputs:tt, $params:tt, $state:tt, $noise:tt,
    { drift ($($drift:tt)*) diffusion ($($diffusion:tt)*) shock ($($shock:tt)*) }) => {
    device_families!(@method lift, $inputs, $params, $state, $noise, $($drift)*, $($diffusion)*, $($shock)*);
  };

  (@history $inputs:tt, $params:tt, $state:tt, $noise:tt,) => {};
  (@history $inputs:tt, $params:tt, $state:tt, $noise:tt,
    { push ($($push:tt)*) weights ($weight:ident) }) => {
    device_families!(@method history, $inputs, $params, $state, $noise, $($push)*);
  };

  (@series $inputs:tt, $params:tt, $state:tt, $noise:tt,) => {};
  (@series $inputs:tt, $params:tt, $state:tt, $noise:tt, { size ($($size:tt)*) }) => {
    device_families!(@method series, $inputs, $params, $state, $noise, $($size)*);
  };
  (@series $inputs:tt, $params:tt, $state:tt, $noise:tt, { live size ($($size:tt)*) }) => {
    device_families!(@method series, $inputs, $params, $state, $noise, $($size)*);
  };

  (@table $inputs:tt, $params:tt, $state:tt, $noise:tt,) => {};
  (@table $inputs:tt, $params:tt, $state:tt, $noise:tt, { increment ($($increment:tt)*) }) => {
    device_families!(@method table, $inputs, $params, $state, $noise, $($increment)*);
  };

  (@method $method:ident, (
    $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident,
    $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $lv:ident, $cv:ident, $sj:ident,
    $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $pv:ident, $pv2:ident,
    state $state_slots:tt, noise $noise_slots:tt, select($component:ident, $produced:ident)
  ), [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*], $($body:tt)*) => {
    #[allow(unused_imports, unused_variables, unused_mut)]
    unsafe fn $method($params: ReadBuffer<f32>, state: [f32; 4], input: Inputs) -> [f32; 4] {
      use super::vocabulary::*;
      device_families!(@inputs input, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7,
        $nj, $js, $gm, $gm2, $u, $u2, $lv, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $pv, $pv2);
      device_families!(@params $params, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19], $($param)*);
      device_families!(@slots state, [0 1 2 3], $($state)*);
      let noise = input.noise;
      device_families!(@slots noise, [0 1 2 3], $($noise)*);
      let mut output = state;
      device_families!(@assign output, [0 1 2 3], $($body)*);
      output
    }
  };

  (@inputs $input:ident,
    $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident,
    $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $lv:ident, $cv:ident, $sj:ident,
    $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $pv:ident, $pv2:ident
  ) => {
    let $dt = $input.dt;
    let $ct = $input.curve[0];
    let $ct1 = $input.curve[1];
    let $ct2 = $input.curve[2];
    let $ct3 = $input.curve[3];
    let $ct4 = $input.curve[4];
    let $ct5 = $input.curve[5];
    let $ct6 = $input.curve[6];
    let $ct7 = $input.curve[7];
    let $nj = $input.count;
    let $js = $input.jump;
    let $gm = $input.gamma[0];
    let $gm2 = $input.gamma[1];
    let $u = $input.uniform[0];
    let $u2 = $input.uniform[1];
    let $lv = $input.lift;
    let $cv = $input.convolution;
    let $sj = $input.series;
    let $gj = $input.arrival;
    let $ej = $input.exponential;
    let $uj = $input.series_uniform[0];
    let $uv = $input.series_uniform[1];
    let $iv = $input.inverse;
    let $tv = $input.spacing;
    let $pv = $input.program[0];
    let $pv2 = $input.program[1];
  };

  (@params $params:ident, [$index:literal $($indices:literal)*], $name:ident $($names:ident)*) => {
    // SAFETY: the frame requires a parameter buffer covering this declaration.
    let $name = unsafe { $params.load_unchecked($index) };
    device_families!(@params $params, [$($indices)*], $($names)*);
  };
  (@params $params:ident, [$($indices:literal)*],) => {};

  (@slots $source:ident, [$index:literal $($indices:literal)*], $name:ident $($names:ident)*) => {
    let $name = $source[$index];
    device_families!(@slots $source, [$($indices)*], $($names)*);
  };
  (@slots $source:ident, [$($indices:literal)*],) => {};

  (@assign $output:ident, $indices:tt, bind $name:ident = $value:expr; $($tail:tt)*) => {
    let $name = $value;
    device_families!(@assign $output, $indices, $($tail)*);
  };
  (@assign $output:ident, [$index:literal $($indices:literal)*], $head:expr, $($tail:expr),+ $(,)?) => {
    $output[$index] = $head;
    device_families!(@assign $output, [$($indices)*], $($tail),+);
  };
  (@assign $output:ident, [$index:literal $($indices:literal)*], $value:expr $(,)?) => {
    $output[$index] = $value;
  };

  (@count bind $name:ident = $value:expr; $($tail:tt)*) => {
    device_families!(@count $($tail)*)
  };
  (@count $($value:expr),* $(,)?) => { [$(stringify!($value)),*].len() as u32 };
}
