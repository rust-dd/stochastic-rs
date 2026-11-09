//! The library crates' `python` feature is internal: every `pub mod python…`, `pyclass` attribute and
//! `pub use` of a `python…` path outside a hidden `python` module is `#[doc(hidden)]`.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

/// Crates whose wrappers all live under a hidden `pub mod python`.
const WRAPPERS_IN_PYTHON_MODULE: [&str; 5] = [
  "stochastic-rs-ai",
  "stochastic-rs-copulas",
  "stochastic-rs-core",
  "stochastic-rs-quant",
  "stochastic-rs-stats",
];

fn library_crates(root: &Path) -> Vec<String> {
  let mut names = fs::read_dir(root)
    .unwrap()
    .map(|entry| entry.unwrap())
    .filter(|entry| entry.path().join("Cargo.toml").is_file())
    .map(|entry| entry.file_name().to_string_lossy().into_owned())
    .filter(|name| name.starts_with("stochastic-rs-") && name != "stochastic-rs-py")
    .collect::<Vec<_>>();
  names.sort();
  names
}

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
  line.trim_start().starts_with("pub mod python")
}

fn is_pyclass_attribute(line: &str) -> bool {
  let line = line.trim_start();
  line.starts_with("#[") && line.contains("pyclass")
}

fn is_python_reexport(line: &str) -> bool {
  let line = line.trim_start();
  line.starts_with("pub use ")
    && line
      .split(|c: char| !(c.is_alphanumeric() || c == '_'))
      .any(|word| word.starts_with("python"))
}

#[test]
fn the_python_surface_of_every_library_crate_is_doc_hidden() {
  let root = Path::new(env!("CARGO_MANIFEST_DIR"));
  let crates = library_crates(root);
  assert!(
    !crates.is_empty(),
    "no stochastic-rs-* library crate next to the umbrella"
  );
  let mut leaks = Vec::new();
  for name in &crates {
    let src = root.join(name).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    files.sort();
    for file in files {
      let text = fs::read_to_string(&file).unwrap();
      let lines = text.lines().collect::<Vec<_>>();
      let exempt =
        WRAPPERS_IN_PYTHON_MODULE.contains(&name.as_str()) && in_python_module_file(&src, &file);
      for (index, line) in lines.iter().enumerate() {
        let exposed = is_public_python_module(line)
          || (!exempt && (is_pyclass_attribute(line) || is_python_reexport(line)));
        if exposed && !hidden_above(&lines, index) {
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
