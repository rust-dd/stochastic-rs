//! What the law suite covers, and what it does not.
//!
//! The August 2026 coverage audit put 77 of 125 sampler types on "runs and is
//! deterministic" and nothing more. A number nobody can see does not move, so
//! this case reads the crate's own sources for the processes that exist,
//! subtracts the ones a law case names, and holds the covered count to a
//! floor that may only be raised. Run it with `--nocapture` to print what is
//! still outstanding.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// The processes a case in this suite states a closed form for. Adding a name
/// here without a case to back it is the one way to make this test lie, so it
/// is the list a reviewer reads first.
const COVERED: &[&str] = &[
  "ARp",
  "Agarch",
  "AlphaStableSubordinator",
  "Arch",
  "Arima",
  "BilateralGamma",
  "BlackKarasinski",
  "Cir",
  "Egarch",
  "GammaSubordinator",
  "Garch",
  "GjrGarch",
  "HoLee",
  "HullWhite",
  "Ig",
  "MAq",
  "Nig",
  "PoissonSubordinator",
  "Sarima",
  "Vasicek",
  "Vg",
];

/// The floor the covered count may not fall below. Raise it with the wave
/// that earns it; never lower it.
const FLOOR: usize = 21;

/// Every `ProcessExt` implementor the crate declares, by type name.
fn declared_processes() -> BTreeSet<String> {
  let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
  let mut found = BTreeSet::new();
  let mut stack = vec![root];
  while let Some(dir) = stack.pop() {
    let entries = fs::read_dir(&dir).expect("the crate's own source tree is readable");
    for entry in entries.flatten() {
      let path = entry.path();
      if path.is_dir() {
        stack.push(path);
        continue;
      }
      if path.extension().is_none_or(|e| e != "rs") {
        continue;
      }
      let text = fs::read_to_string(&path).expect("a source file is readable");
      // The implementation header may wrap, so the name is taken from the
      // `for` that follows the trait rather than from one line of the file —
      // but only while still inside that header, or a mention in a doc
      // comment would claim the next `impl` in the file as a process.
      for (offset, _) in text.match_indices("ProcessExt<T>") {
        // The window is trimmed back to a character boundary: these
        // sources carry σ and θ in their doc comments.
        let mut end = (offset + 200).min(text.len());
        while !text.is_char_boundary(end) {
          end -= 1;
        }
        let tail = &text[offset..end];
        let Some(at) = tail.find(" for ") else {
          continue;
        };
        if tail[..at].contains(['{', ';']) || tail[..at].contains("\n\n") {
          continue;
        }
        let name: String = tail[at + 5..]
          .chars()
          .take_while(|c| c.is_alphanumeric() || *c == '_')
          .collect();
        if !name.is_empty() && name.chars().next().is_some_and(char::is_uppercase) {
          found.insert(name);
        }
      }
    }
  }
  found
}

/// The covered set only grows, and every name in it is a process the crate
/// actually declares — a renamed process must not quietly leave the list
/// looking full.
#[test]
fn the_law_suite_covers_what_it_claims() {
  let declared = declared_processes();
  assert!(
    declared.len() > 100,
    "only {} processes found; the source scan is broken, not the coverage",
    declared.len()
  );
  let covered: BTreeSet<String> = COVERED.iter().map(|s| (*s).to_string()).collect();
  let unknown: Vec<&String> = covered.difference(&declared).collect();
  assert!(
    unknown.is_empty(),
    "these names are claimed as covered but are not processes of this crate: {unknown:?}"
  );
  assert!(
    covered.len() >= FLOOR,
    "the law suite covers {} processes, below the floor of {FLOOR}",
    covered.len()
  );
  let outstanding: Vec<&String> = declared.difference(&covered).collect();
  println!(
    "law coverage: {} of {} processes; outstanding: {outstanding:?}",
    covered.len(),
    declared.len()
  );
}
