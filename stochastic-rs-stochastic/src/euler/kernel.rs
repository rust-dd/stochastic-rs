//! CUDA Euler kernels specialized from the shared family declarations.

mod frame;

mod language;

#[cfg(test)]
mod tests;

use frame::{FRAME_BLOCKS, Need};
use language::substitute;
pub(crate) use language::{Language, cuda_language, prelude};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Shape {
  pub(crate) family: u32,
  pub(crate) lift: bool,
  pub(crate) history: bool,
  pub(crate) series: bool,
  pub(crate) table: bool,
  pub(crate) program: bool,
  pub(crate) jumps: bool,
  pub(crate) uniforms: bool,
  pub(crate) noises: u8,
  pub(crate) components: u8,
  pub(crate) gamma: bool,
}

// Match whole identifiers: `u` inside `unsigned` or `uj` is not a uniform read.
fn reads(block: &str, name: &str) -> bool {
  let bytes = block.as_bytes();
  let mut from = 0;
  while let Some(at) = block[from..].find(name) {
    let start = from + at;
    let end = start + name.len();
    let before = start
      .checked_sub(1)
      .map(|i| bytes[i])
      .is_none_or(|c| !c.is_ascii_alphanumeric() && c != b'_');
    let after = bytes
      .get(end)
      .is_none_or(|c| !c.is_ascii_alphanumeric() && *c != b'_');
    if before && after {
      return true;
    }
    from = end;
  }
  false
}

impl Shape {
  pub(crate) fn new(family: super::families::Family, jumps: bool, gamma: bool) -> Self {
    use super::families;
    Self {
      family: family.code(),
      lift: !families::c_lift_for(family).is_empty(),
      history: !families::c_history_for(family).is_empty(),
      series: !families::c_series_for(family).is_empty(),
      table: !families::c_table_for(family).is_empty(),
      // A program value can be read anywhere the family writes C: the
      // Volterra equation takes both of them in its lift, not its step.
      program: [
        families::c_step_for(family),
        families::c_lift_for(family),
        families::c_report_for(family),
        families::c_series_for(family),
        families::c_table_for(family),
        families::c_history_for(family),
      ]
      .iter()
      .any(|block| block.contains("pv")),
      jumps,
      gamma,
      uniforms: [
        families::c_step_for(family),
        families::c_lift_for(family),
        families::c_report_for(family),
        families::c_history_for(family),
        families::c_series_for(family),
        families::c_table_for(family),
      ]
      .iter()
      .any(|block| reads(block, "u") || reads(block, "u2")),
      noises: family.noises() as u8,
      components: family.components() as u8,
    }
  }

  fn needs_block(&self) -> bool {
    self.history || self.series || self.table
  }
}

pub(crate) fn render_for(lang: &Language<'_>, shape: Shape) -> String {
  use super::families;
  let family = families::Family::from_code(shape.family).expect("a declared family");
  let keep = |need: Need| -> bool {
    match need {
      Need::Always => true,
      Need::Series => shape.series,
      Need::Table => shape.table,
      Need::Program => shape.program,
      Need::Lift => shape.lift,
      Need::History => shape.history,
      Need::Counts => shape.jumps || shape.gamma,
      Need::Jumps => shape.jumps,
      Need::Uniforms => shape.uniforms,
    }
  };
  let mut body = FRAME_BLOCKS
    .iter()
    .filter(|(_, need)| keep(*need))
    .map(|(block, _)| *block)
    .collect::<Vec<_>>()
    .join("\n");
  if !shape.lift {
    body = body
      .replace("REAL lh[176];", "REAL lh[1];")
      .replace("REAL lj[176];", "REAL lj[1];");
  }
  if !shape.needs_block() {
    body = body.replace("REAL block[512];", "REAL block[1];");
  }
  // Separate scopes prevent the step and report parameter bindings from colliding.
  let scoped = |body: &str, indent: &str| -> String {
    if body.trim().is_empty() {
      return String::new();
    }
    format!("{indent}{{\n{}\n{indent}}}", body.trim_end_matches('\n'))
  };
  let body = body
    .replace("NOISES", &format!("{}u", shape.noises))
    .replace("COMPONENTS", &format!("{}u", shape.components))
    .replace(
      "SERIES",
      &scoped(families::c_series_for(family), "            "),
    )
    .replace(
      "TABLE",
      &scoped(families::c_table_for(family), "                "),
    )
    .replace("LIFT", &scoped(families::c_lift_for(family), "        "))
    .replace(
      "HISTORY",
      &scoped(families::c_history_for(family), "        "),
    )
    .replace("STEP", &scoped(families::c_step_for(family), "        "))
    .replace(
      "REPORT",
      &scoped(families::c_report_for(family), "        "),
    );
  substitute(&body, lang)
}

#[cfg(test)]
pub(crate) fn render(lang: &Language<'_>) -> String {
  let body = FRAME_BLOCKS
    .iter()
    .map(|(block, _)| *block)
    .collect::<Vec<_>>()
    .join("\n")
    .replace("NOISES", "noises")
    .replace("COMPONENTS", "components")
    .replace("SERIES", super::families::C_SERIES.trim_end_matches('\n'))
    .replace("TABLE", super::families::C_TABLE.trim_end_matches('\n'))
    .replace("LIFT", super::families::C_LIFT.trim_end_matches('\n'))
    .replace("HISTORY", super::families::C_HISTORY.trim_end_matches('\n'))
    .replace("STEP", super::families::C_STEP.trim_end_matches('\n'))
    .replace("REPORT", super::families::C_REPORT.trim_end_matches('\n'));
  substitute(&body, lang)
}
