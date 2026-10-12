//! With the `metal` feature, keeps the Euler kernel entries and the embedded Metal artifact
//! in step with the kernel sources. Packages without the kernel crate use the artifact as is.

fn main() -> Result<(), Box<dyn std::error::Error>> {
  println!("cargo:rerun-if-changed=build.rs");
  #[cfg(feature = "metal")]
  {
    entries::update(
      std::path::Path::new("src/euler/families/definitions"),
      std::path::Path::new("../stochastic-rs-metal-kernels/src/euler/entries"),
    )?;
    metal_oxide_build::Artifact::new(
      "../stochastic-rs-metal-kernels/Cargo.toml",
      "src/device/metal/artifact",
    )
    .math_mode(metal_oxide_build::MathMode::Fast)
    .update()?;
  }
  Ok(())
}

/// One kernel entry per Euler family and jump/gamma shape that changes the draws it reads.
#[cfg(feature = "metal")]
mod entries {
  use std::collections::BTreeMap;
  use std::path::Path;

  type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

  struct Family {
    code: u32,
    name: String,
    jumps: bool,
    gamma: bool,
  }

  /// Rewrites the entry files in `destination` when the families in `definitions` changed.
  /// `METAL_OXIDE_ARTIFACTS=locked` makes stale entries an error instead.
  pub fn update(definitions: &Path, destination: &Path) -> Result<()> {
    println!("cargo:rerun-if-changed={}", definitions.display());
    if !destination.is_dir() {
      return Ok(());
    }
    let expected = sources(definitions)?;
    let mut stale = Vec::new();
    for entry in std::fs::read_dir(destination)? {
      let name = entry?.file_name().to_string_lossy().into_owned();
      if name.ends_with(".rs") && !expected.contains_key(&name) {
        stale.push(name);
      }
    }
    let changed = expected
      .iter()
      .filter(|(name, contents)| {
        std::fs::read_to_string(destination.join(name))
          .ok()
          .as_ref()
          != Some(contents)
      })
      .collect::<Vec<_>>();
    if stale.is_empty() && changed.is_empty() {
      return Ok(());
    }
    if std::env::var("METAL_OXIDE_ARTIFACTS").as_deref() == Ok("locked") {
      return Err(format!("Euler entries in {} are stale", destination.display()).into());
    }
    for name in stale {
      std::fs::remove_file(destination.join(name))?;
    }
    for (name, contents) in changed {
      std::fs::write(destination.join(name), contents)?;
    }
    Ok(())
  }

  fn sources(definitions: &Path) -> Result<BTreeMap<String, String>> {
    let families = families(definitions)?;
    let mut files = BTreeMap::new();
    let mut modules = Vec::new();
    for chunk in families.chunks(32) {
      let module = format!("families_{}_{}", chunk[0].code, chunk[chunk.len() - 1].code);
      modules.push(format!("mod {module};\npub use {module}::*;"));
      let mut lines = Vec::new();
      for family in chunk {
        for jumps in [false, true]
          .into_iter()
          .take(1 + usize::from(family.jumps))
        {
          for gamma in [false, true]
            .into_iter()
            .take(1 + usize::from(family.gamma))
          {
            lines.push(format!(
              "entrypoint!(euler_{}_{}_{}, {}, {jumps}, {gamma});",
              family.code,
              u8::from(jumps),
              u8::from(gamma),
              family.name
            ));
          }
        }
      }
      files.insert(format!("{module}.rs"), lines.join("\n") + "\n");
    }
    files.insert("mod.rs".into(), modules.join("\n") + "\n");
    Ok(files)
  }

  /// Each `code => Name {` declaration, ordered by code, which must be dense from zero.
  fn families(definitions: &Path) -> Result<Vec<Family>> {
    let mut paths = std::fs::read_dir(definitions)?
      .map(|entry| entry.map(|entry| entry.path()))
      .collect::<std::io::Result<Vec<_>>>()?;
    paths.retain(|path| path.extension() == Some("rs".as_ref()));
    paths.sort();
    let mut families = Vec::new();
    for path in paths {
      let text = std::fs::read_to_string(&path)?;
      let headers = headers(&text);
      for (index, (_, end, code, name)) in headers.iter().enumerate() {
        let next = headers.get(index + 1).map_or(text.len(), |header| header.0);
        let (jumps, gamma) = draws(&text[*end..next])
          .ok_or_else(|| format!("family {name} in {} has no noise list", path.display()))?;
        families.push(Family {
          code: *code,
          name: name.clone(),
          jumps,
          gamma,
        });
      }
    }
    families.sort_by_key(|family| family.code);
    if families.is_empty()
      || families
        .iter()
        .enumerate()
        .any(|(index, family)| family.code as usize != index)
    {
      return Err("family codes must be unique, dense and start at zero".into());
    }
    let mut names = families
      .iter()
      .map(|family| &family.name)
      .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    if names.len() != families.len() {
      return Err("family names must be unique".into());
    }
    Ok(families)
  }

  /// `(line start, end of the opening brace, code, name)` for each `code => Name {` line.
  fn headers(text: &str) -> Vec<(usize, usize, u32, String)> {
    let mut headers = Vec::new();
    let mut start = 0;
    for line in text.split_inclusive('\n') {
      let rest = line.trim_start();
      let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
      let parsed = (digits > 0)
        .then(|| rest[digits..].trim_start().strip_prefix("=>"))
        .flatten()
        .and_then(|rest| {
          let rest = rest.trim_start();
          let name = rest.len()
            - rest
              .trim_start_matches(|c: char| c.is_ascii_alphanumeric() || c == '_')
              .len();
          let first = rest.chars().next()?;
          let brace = rest[name..].trim_start().strip_prefix('{')?;
          (name > 0 && !first.is_ascii_digit()).then(|| (rest[..name].to_owned(), brace))
        });
      if let Some((name, brace)) = parsed {
        let code = rest[..digits].parse().unwrap_or(u32::MAX);
        headers.push((start, start + line.len() - brace.len(), code, name));
      }
      start += line.len();
    }
    headers
  }

  /// Whether the clauses after a family's `noise (...)` list read jump (`nj`, `js`) or
  /// gamma (`gm`, `gm2`) draws; gamma draws imply the jump variant.
  fn draws(declaration: &str) -> Option<(bool, bool)> {
    let code = declaration
      .split_inclusive('\n')
      .map(|line| line.find("//").map_or(line, |comment| &line[..comment]))
      .collect::<String>();
    let clauses = code[noise(&code)?..].replace('$', "");
    let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let reads = |name: &str| {
      clauses.match_indices(name).any(|(index, _)| {
        !clauses[..index].chars().next_back().is_some_and(word)
          && !clauses[index + name.len()..]
            .chars()
            .next()
            .is_some_and(word)
      })
    };
    let gamma = reads("gm") || reads("gm2");
    Some((gamma || reads("nj") || reads("js"), gamma))
  }

  /// The end of the first `noise (...)` list.
  fn noise(code: &str) -> Option<usize> {
    let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    code.match_indices("noise").find_map(|(index, _)| {
      if code[..index].chars().next_back().is_some_and(word) {
        return None;
      }
      let open = code[index + "noise".len()..]
        .trim_start()
        .strip_prefix('(')?;
      let close = open.find(')')?;
      Some(code.len() - open.len() + close + 1)
    })
  }
}
