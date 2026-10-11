//! CUDA renderer coverage without a device or driver compiler.

use super::cuda_language;
use super::prelude;
use super::render;

fn languages() -> Vec<(&'static str, super::Language<'static>)> {
  vec![
    ("CUDA f32", cuda_language("float")),
    ("CUDA f64", cuda_language("double")),
  ]
}

#[test]
fn every_family_renders_a_kernel_of_its_own() {
  for (name, lang) in languages() {
    for family in super::super::families::Family::ALL {
      let shape = super::Shape::new(*family, true, true);
      let source = super::render_for(&lang, shape);
      assert!(
        !source.contains("if (family =="),
        "{name}/{family:?}: a specialised kernel still compares the family"
      );
      for placeholder in [
        "STOCH_", "INDEX", "U64", "STEP", "REPORT", "LIFT", "HISTORY",
      ] {
        assert!(
          !source.contains(placeholder),
          "{name}/{family:?}: the {placeholder} placeholder survived"
        );
      }
      let balance = source.chars().fold(0i32, |d, c| match c {
        '{' => d + 1,
        '}' => d - 1,
        _ => d,
      });
      assert_eq!(balance, 0, "{name}/{family:?}: unbalanced body");
    }
  }
}

#[test]
fn a_plain_family_declares_no_scratch_it_cannot_reach() {
  let lang = cuda_language("float");
  let plain = super::Shape::new(
    super::super::families::Family::GeometricBrownian,
    false,
    false,
  );
  let source = super::render_for(&lang, plain);
  for scratch in ["lh[176]", "lj[176]", "block[512]"] {
    assert!(
      !source.contains(scratch),
      "a plain family still declares {scratch}"
    );
  }
  assert!(
    !source.contains("jump_law == 9u") && !source.contains("gi < gamma_law"),
    "a launch with no jumps still carries the jump laws"
  );
  let full = super::render_for(
    &lang,
    super::Shape::new(
      super::super::families::Family::GeometricBrownian,
      true,
      true,
    ),
  );
  assert!(
    full.len() > source.len(),
    "the jump laws must cost something, or dropping them buys nothing"
  );
}

#[test]
fn the_frame_blocks_join_into_a_balanced_body() {
  let joined = super::FRAME_BLOCKS
    .iter()
    .map(|(block, _)| *block)
    .collect::<Vec<_>>()
    .join("\n");
  let mut depth = 0i32;
  let mut lowest = 0i32;
  for c in joined.chars() {
    match c {
      '{' => depth += 1,
      '}' => depth -= 1,
      _ => {}
    }
    lowest = lowest.min(depth);
  }
  assert_eq!(
    depth, 0,
    "the frame opens and closes its own braces — the kernel's function brace \
     belongs to the language header and its close is appended by the \
     back-end — and this body is left at {depth}"
  );
  assert_eq!(
    lowest, 0,
    "a block closes a brace no earlier block opened; the order is wrong"
  );
  // Optional fragments close their braces; the required pair frames the step loop.
  for (i, (block, _)) in super::FRAME_BLOCKS.iter().enumerate() {
    let balance = block.chars().fold(0i32, |d, c| match c {
      '{' => d + 1,
      '}' => d - 1,
      _ => d,
    });
    // The two that frame the step loop, by their place in the list.
    let want = match i {
      4 => 1,
      13 => -1,
      _ => 0,
    };
    assert_eq!(
      balance, want,
      "block {i} leaves {balance} braces open, not {want}; an optional block \
       that does not close what it opens cannot be left out"
    );
  }
}

#[test]
fn every_placeholder_is_substituted() {
  for (name, lang) in languages() {
    let source = format!("{}{}", prelude(&lang), render(&lang));
    assert!(
      !source.contains("STOCH_"),
      "{name}: an intrinsic placeholder survived rendering — \
       the vocabulary names it but this Language table does not"
    );
    assert!(
      !source.contains("INDEX"),
      "{name}: the buffer-index placeholder survived rendering"
    );
    assert!(
      !source.contains("U64"),
      "{name}: the wide-integer placeholder survived rendering"
    );
  }
}

#[test]
fn the_cell_key_takes_the_path_and_the_step_apart() {
  for (name, lang) in languages() {
    let source = render(&lang);
    assert!(
      source.contains("unsigned int g = (first_path + path) * 2654435761u;"),
      "{name}: the path does not enter the key through an odd multiplier"
    );
    assert!(
      !source.contains("(first_path + path) * steps"),
      "{name}: the key is a product of the path and the step count"
    );
  }
}

#[test]
fn every_intrinsic_is_named() {
  for (name, lang) in languages() {
    for (what, value) in [
      ("real", lang.real),
      ("sqrt", lang.sqrt),
      ("log", lang.log),
      ("cos", lang.cos),
      ("sin", lang.sin),
      ("exp", lang.exp),
      ("pow", lang.pow),
      ("abs", lang.abs),
      ("tanh", lang.tanh),
      ("atan", lang.atan),
      ("index", lang.index),
    ] {
      assert!(!value.is_empty(), "{name}: `{what}` has no intrinsic");
    }
  }
}

#[test]
fn every_family_reaches_the_rendered_kernel() {
  let lang = cuda_language("float");
  let body = render(&lang);
  for family in super::super::families::Family::ALL {
    let code = family.code();
    assert!(
      body.contains(&format!("family == {code}u")),
      "{family:?} (code {code}) is declared but the rendered kernel never \
       dispatches to it"
    );
  }
}
