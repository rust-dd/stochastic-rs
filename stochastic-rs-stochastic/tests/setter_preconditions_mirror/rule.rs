use std::collections::HashSet;
use std::path::Path;

use crate::Stmt;
use crate::mask;
use crate::parse_impls;
use crate::statements;

/// (type, setter, why) pairs the mirror rule does not cover.
const ALLOWED: &[(&str, &str, &str)] = &[
  (
    "Fgn",
    "with_hurst",
    "rebuilds through `Self::new`, which asserts",
  ),
  (
    "Fgn",
    "with_increments",
    "rebuilds through `Self::new`, which asserts",
  ),
  (
    "Fgn",
    "with_horizon",
    "rebuilds through `Self::new`, which asserts",
  ),
  (
    "Fbs",
    "with_rows",
    "rebuilds through `Self::new`, which asserts",
  ),
  (
    "Fbs",
    "with_cols",
    "rebuilds through `Self::new`, which asserts",
  ),
  (
    "Lmm",
    "with_correlation",
    "`new` takes the matrix `rho` under another check",
  ),
];

struct Setter {
  ty: String,
  name: String,
  arg: String,
  have: Vec<String>,
  at: String,
}

struct News {
  params: Vec<String>,
  stmts: Vec<Stmt>,
}

fn check(files: &[(String, String)]) -> (Vec<String>, usize, usize) {
  let mut news: Vec<(String, News)> = vec![];
  let mut setters: Vec<Setter> = vec![];
  let mut skipped: Vec<(String, String, String)> = vec![];
  for (path, src) in files {
    let m = mask(src);
    for im in parse_impls(src, &m) {
      for f in &im.funs {
        let stmts = statements(src, &m, f.body.0, f.body.1);
        if f.name == "new" {
          let params = f.params.iter().map(|p| p.0.clone()).collect();
          news.push((im.ty.clone(), News { params, stmts }));
        } else if f.public && f.name.starts_with("with_") && f.params.len() == 1 {
          setters.push(Setter {
            ty: im.ty.clone(),
            name: f.name.clone(),
            arg: f.params[0].0.clone(),
            have: stmts
              .into_iter()
              .map(|s| s.text)
              .filter(|t| !t.contains("self."))
              .collect(),
            at: path.clone(),
          });
        }
      }
    }
  }
  let mut errors = vec![];
  let mut asserting = 0;
  for s in &setters {
    if let Some((_, _, why)) = ALLOWED.iter().find(|(t, f, _)| *t == s.ty && *f == s.name) {
      skipped.push((s.ty.clone(), s.name.clone(), (*why).to_string()));
      continue;
    }
    let cands: Vec<&News> = news
      .iter()
      .filter(|(t, _)| *t == s.ty)
      .map(|(_, n)| n)
      .collect();
    if cands.is_empty() {
      errors.push(format!(
        "{}::{} ({}): the type has no `new`",
        s.ty, s.name, s.at
      ));
      continue;
    }
    let ok = cands.iter().any(|n| {
      let want: Vec<&String> = n
        .stmts
        .iter()
        .filter(|st| {
          let used: HashSet<&String> = st.names.iter().filter(|x| n.params.contains(x)).collect();
          used.len() == 1 && used.contains(&s.arg)
        })
        .map(|st| &st.text)
        .collect();
      want.len() == s.have.len() && want.iter().zip(&s.have).all(|(a, b)| *a == b)
    });
    if !s.have.is_empty() {
      asserting += 1;
    }
    let own = |t: &&String| t.starts_with("assert!(") || t.starts_with("assert_entries(");
    if let Some(own) = s.have.iter().find(own) {
      errors.push(format!(
        "{}::{} ({}): `{own}` belongs in the `check_{}` that `new` calls too",
        s.ty, s.name, s.at, s.arg
      ));
      continue;
    }
    let taken = cands.iter().any(|n| n.params.contains(&s.arg));
    let named = format!("Self::check_{0}({0});", s.arg);
    if !taken && !s.have.is_empty() && s.have != [named.clone()] {
      errors.push(format!(
        "{}::{} ({}): `new` does not take `{}`, so the setter runs `{named}`, not {:?}",
        s.ty, s.name, s.at, s.arg, s.have
      ));
      continue;
    }
    if taken && !ok {
      let want = cands
        .iter()
        .map(|n| {
          n.stmts
            .iter()
            .filter(|st| {
              let used: HashSet<&String> =
                st.names.iter().filter(|x| n.params.contains(x)).collect();
              used.len() == 1 && used.contains(&s.arg)
            })
            .map(|st| st.text.clone())
            .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
      errors.push(format!(
        "{}::{} ({}): setter asserts {:?}, `new` asserts {:?} on `{}`",
        s.ty, s.name, s.at, s.have, want, s.arg
      ));
    }
  }
  let stale: Vec<String> = ALLOWED
    .iter()
    .filter(|(t, f, _)| !skipped.iter().any(|(st, sf, _)| st == t && sf == f))
    .map(|(t, f, _)| format!("allow-list entry {t}::{f} matches no setter"))
    .collect();
  errors.extend(stale);
  (errors, setters.len(), asserting)
}

fn read_tree(dir: &Path, out: &mut Vec<(String, String)>) {
  let mut entries: Vec<_> = std::fs::read_dir(dir)
    .unwrap()
    .map(|e| e.unwrap().path())
    .collect();
  entries.sort();
  for p in entries {
    if p.is_dir() {
      read_tree(&p, out);
    } else if p.extension().is_some_and(|e| e == "rs") {
      out.push((
        p.display().to_string(),
        std::fs::read_to_string(&p).unwrap(),
      ));
    }
  }
}

#[test]
fn every_setter_asserts_what_new_asserts() {
  let mut files = vec![];
  read_tree(
    &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
    &mut files,
  );
  let (errors, setters, asserting) = check(&files);
  assert!(
    setters > 350 && asserting > 300,
    "parsed {setters} setters, {asserting} asserting"
  );
  assert!(
    errors.is_empty(),
    "{} setters drifted from `new`:\n{}",
    errors.len(),
    errors.join("\n")
  );
}

const MIRROR: &str = r#"
impl<T: FloatExt> Foo<T> {
  pub fn new(sigma: T, n: usize) -> Self {
    Self::check_sigma(sigma);
    check_n(n);
    Self { sigma, n, mu: T::zero() }
  }
}

impl<T: FloatExt> Foo<T> {
  fn check_sigma(sigma: T) {
    require!(sigma > T::zero(), sigma);
  }

  fn check_mu(mu: T) {
    require!(mu.is_finite(), mu);
  }

  pub fn with_sigma(mut self, sigma: T) -> Self {
    Self::check_sigma(sigma);
    self.sigma = sigma;
    self
  }

  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self
  }

  pub fn with_mu(mut self, mu: T) -> Self {
    Self::check_mu(mu);
    self.mu = mu;
    self
  }
}
"#;

#[test]
fn a_mirrored_setter_passes() {
  let (errors, setters, asserting) = check(&[("mirror.rs".into(), MIRROR.into())]);
  assert_eq!((setters, asserting), (3, 3));
  assert!(
    errors.iter().all(|e| e.starts_with("allow-list entry")),
    "{errors:?}"
  );
}

#[test]
fn a_setter_with_its_own_assert_or_without_the_check_fails() {
  let (shared, own) = (
    "    check_n(n);\n    self.n",
    "    assert!(n >= 2, \"n must satisfy `n >= 2`, got n = {n:?}\");\n    self.n",
  );
  let (errors, ..) = check(&[("own.rs".into(), MIRROR.replacen(shared, own, 1))]);
  assert!(
    errors.iter().any(|e| e.starts_with("Foo::with_points")),
    "{errors:?}"
  );
  let dropped = MIRROR.replacen(
    "    Self::check_sigma(sigma);\n    self.sigma",
    "    self.sigma",
    1,
  );
  let (errors, ..) = check(&[("dropped.rs".into(), dropped.clone())]);
  assert!(
    errors.iter().any(|e| e.starts_with("Foo::with_sigma")),
    "{errors:?}"
  );
  let (errors, ..) = check(&[("bare.rs".into(), MIRROR.replacen(shared, "    self.n", 1))]);
  assert!(
    errors.iter().any(|e| e.starts_with("Foo::with_points")),
    "{errors:?}"
  );
  let (in_new, written) = (
    "    Self::check_sigma(sigma);\n    check_n",
    "    assert_entries(\"sigma\", \"sigma[i] > 0.0\", &sigma, |v| v > 0.0);\n    check_n",
  );
  let (errors, ..) = check(&[("array.rs".into(), dropped.replacen(in_new, written, 1))]);
  assert!(
    errors.iter().any(|e| e.starts_with("Foo::with_sigma")),
    "{errors:?}"
  );
  let own = "    Self::check_mu(mu);\n    self.mu";
  let written = "    assert_entries(\"mu\", \"mu[i] > 0.0\", &mu, |v| v > 0.0);\n    self.mu";
  let (errors, ..) = check(&[("own.rs".into(), MIRROR.replacen(own, written, 1))]);
  assert!(
    errors.iter().any(|e| e.starts_with("Foo::with_mu")),
    "{errors:?}"
  );
  let other = "    Self::check_sigma(mu);\n    self.mu";
  let (errors, ..) = check(&[("other.rs".into(), MIRROR.replacen(own, other, 1))]);
  assert!(
    errors.iter().any(|e| e.starts_with("Foo::with_mu")),
    "{errors:?}"
  );
}
