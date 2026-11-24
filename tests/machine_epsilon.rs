//! Generic code reads a scalar's epsilon from `RealExt::machine_epsilon`, which every scalar must
//! give: no library crate calls `::epsilon()`, whose num-traits default is `f32`'s.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

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

#[test]
fn no_library_source_calls_float_epsilon() {
  let root = Path::new(env!("CARGO_MANIFEST_DIR"));
  let mut files = Vec::new();
  rust_files(&root.join("src"), &mut files);
  for entry in fs::read_dir(root).unwrap() {
    let path = entry.unwrap().path();
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    if name.starts_with("stochastic-rs-") && path.join("src").is_dir() {
      rust_files(&path.join("src"), &mut files);
    }
  }
  assert!(files.len() > 100, "found {} source files", files.len());
  let calls = files
    .iter()
    .flat_map(|file| {
      let text = fs::read_to_string(file).unwrap();
      text
        .lines()
        .enumerate()
        .filter(|(_, line)| line.contains("::epsilon()"))
        .map(|(index, line)| format!("{}:{}: {}", file.display(), index + 1, line.trim()))
        .collect::<Vec<_>>()
    })
    .collect::<Vec<_>>();
  assert!(calls.is_empty(), "{} calls: {calls:#?}", calls.len());
}
