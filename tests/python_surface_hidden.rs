//! The library crates' `python` feature is internal: every `pub mod python` and every
//! `#[pyclass]` outside such a module is `#[doc(hidden)]`, so no wrapper class is documented API.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

const LIBRARY_CRATES: [&str; 7] = [
  "stochastic-rs-ai",
  "stochastic-rs-copulas",
  "stochastic-rs-core",
  "stochastic-rs-distributions",
  "stochastic-rs-quant",
  "stochastic-rs-stats",
  "stochastic-rs-stochastic",
];

/// Crates whose wrappers all live under a hidden `pub mod python`.
const WRAPPERS_IN_PYTHON_MODULE: [&str; 5] = [
  "stochastic-rs-ai",
  "stochastic-rs-copulas",
  "stochastic-rs-core",
  "stochastic-rs-quant",
  "stochastic-rs-stats",
];

fn rust_files(dir: &Path, files: &mut Vec<PathBuf>) {
  for entry in fs::read_dir(dir).unwrap() {
    let path = entry.unwrap().path();
    if path.is_dir() {
      rust_files(&path, files);
    } else if path.extension().is_some_and(|extension| extension == "rs") {
      files.push(path);
    }
  }
}

fn in_python_module_file(src: &Path, file: &Path) -> bool {
  let relative = file.strip_prefix(src).unwrap();
  relative.starts_with("python") || relative == Path::new("python.rs")
}

fn hidden_above(lines: &[&str], index: usize) -> bool {
  index > 0 && lines[index - 1].trim() == "#[doc(hidden)]"
}

fn is_public_python_module(line: &str) -> bool {
  let line = line.trim_start();
  line.starts_with("pub mod python ")
    || line.starts_with("pub mod python;")
    || line.starts_with("pub mod python_device")
}

fn is_pyclass(line: &str) -> bool {
  let line = line.trim_start();
  line.starts_with("#[pyclass") || line.starts_with("#[pyo3::prelude::pyclass")
}

#[test]
fn the_python_surface_of_every_library_crate_is_doc_hidden() {
  let root = Path::new(env!("CARGO_MANIFEST_DIR"));
  let mut leaks = Vec::new();
  for name in LIBRARY_CRATES {
    let src = root.join(name).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    files.sort();
    for file in files {
      let text = fs::read_to_string(&file).unwrap();
      let lines = text.lines().collect::<Vec<_>>();
      let exempt = WRAPPERS_IN_PYTHON_MODULE.contains(&name) && in_python_module_file(&src, &file);
      for (index, line) in lines.iter().enumerate() {
        let leaked = (is_public_python_module(line) || (is_pyclass(line) && !exempt))
          && !hidden_above(&lines, index);
        if leaked {
          let relative = file.strip_prefix(root).unwrap().display();
          leaks.push(format!("{relative}:{}", index + 1));
        }
      }
    }
  }
  assert!(
    leaks.is_empty(),
    "Python surface without #[doc(hidden)]:\n{}",
    leaks.join("\n")
  );
}
