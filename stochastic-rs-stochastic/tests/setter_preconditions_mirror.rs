//! Every `with_*` setter runs the check its type's `new` runs on the same argument: the field's one
//! `check_*` or the argument's own `assert_*`, and no assert of its own unless on `self.`.

use std::collections::HashSet;
use std::path::Path;

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

const KEYWORDS: &[&str] = &[
  "self", "Self", "true", "false", "as", "if", "else", "let", "mut", "move", "in", "for", "match",
  "return", "where", "fn", "impl", "T", "None", "Some",
];

/// A check call: a type's own, a shared crate function, or the array form written out.
const CHECKS: &[&[u8]] = &[b"Self::check_", b"check_", b"assert_entries("];

fn is_ident(b: u8) -> bool {
  b.is_ascii_alphanumeric() || b == b'_'
}

fn mask(src: &str) -> Vec<u8> {
  let s = src.as_bytes();
  let mut out = s.to_vec();
  let n = s.len();
  let mut i = 0;
  while i < n {
    if s[i..].starts_with(b"//") {
      let j = s[i..].iter().position(|&c| c == b'\n').map_or(n, |p| i + p);
      out[i..j].fill(b' ');
      i = j;
    } else if s[i..].starts_with(b"/*") {
      let j = src[i + 2..].find("*/").map_or(n, |p| i + 2 + p + 2);
      for c in &mut out[i..j] {
        if *c != b'\n' {
          *c = b' ';
        }
      }
      i = j;
    } else if s[i] == b'"' {
      let mut j = i + 1;
      while j < n && s[j] != b'"' {
        if s[j] == b'\\' {
          out[j] = b'x';
          j += 1;
        }
        if j < n && s[j] != b'\n' {
          out[j] = b'x';
        }
        j += 1;
      }
      i = j + 1;
    } else if s[i] == b'\'' {
      let len = if i + 3 < n && s[i + 1] == b'\\' && s[i + 3] == b'\'' {
        4
      } else if i + 2 < n && s[i + 1] != b'\\' && s[i + 1] != b'\'' && s[i + 2] == b'\'' {
        3
      } else {
        0
      };
      if len > 0 {
        out[i + 1..i + len - 1].fill(b'x');
        i += len;
      } else {
        i += 1;
      }
    } else {
      i += 1;
    }
  }
  out
}

fn close(m: &[u8], mut i: usize, open: u8, shut: u8) -> usize {
  let mut d = 1;
  while i < m.len() && d > 0 {
    if m[i] == open {
      d += 1;
    } else if m[i] == shut {
      d -= 1;
    }
    i += 1;
  }
  i
}

fn split_top(s: &str, angle: bool) -> Vec<String> {
  let mut out = vec![];
  let mut cur = String::new();
  let mut d = 0i32;
  for ch in s.chars() {
    match ch {
      '(' | '[' | '{' => d += 1,
      ')' | ']' | '}' => d -= 1,
      '<' if angle => d += 1,
      '>' if angle => d -= 1,
      _ => {}
    }
    if ch == ',' && d == 0 {
      out.push(std::mem::take(&mut cur));
    } else {
      cur.push(ch);
    }
  }
  if !cur.trim().is_empty() {
    out.push(cur);
  }
  out
}

fn squash(src: &str, msk: &[u8]) -> String {
  let s: String = src
    .bytes()
    .zip(msk)
    .filter(|(a, b)| !(a.is_ascii_whitespace() && **b != b'x'))
    .map(|(a, _)| a as char)
    .collect();
  s.replace(",)", ")")
}

fn cut_closures(code: &str) -> String {
  let c = code.as_bytes();
  let mut out = String::new();
  let mut i = 0;
  while i < c.len() {
    if c[i..].starts_with(b"||") {
      out.push_str("||");
      i += 2;
    } else if c[i] == b'|' {
      let mut j = c[i + 1..]
        .iter()
        .position(|&x| x == b'|')
        .map_or(c.len(), |p| i + 2 + p);
      let mut d = 0i32;
      while j < c.len() && !(d == 0 && b",)]}".contains(&c[j])) {
        d += i32::from(b"([{".contains(&c[j])) - i32::from(b")]}".contains(&c[j]));
        j += 1;
      }
      i = j;
    } else {
      out.push(c[i] as char);
      i += 1;
    }
  }
  out
}

fn idents(masked: &str) -> HashSet<String> {
  let code = cut_closures(masked);
  let b = code.as_bytes();
  let mut names = HashSet::new();
  let mut i = 0;
  while i < b.len() {
    if b[i].is_ascii_alphabetic() || b[i] == b'_' {
      let s = i;
      while i < b.len() && is_ident(b[i]) {
        i += 1;
      }
      let w = &code[s..i];
      let after_path = b[i..].starts_with(b"::");
      let before_path = s >= 1 && (b[s - 1] == b'.' || (s >= 2 && &b[s - 2..s] == b"::"));
      if !(after_path
        || before_path
        || KEYWORDS.contains(&w)
        || w.as_bytes()[0].is_ascii_uppercase())
      {
        names.insert(w.to_string());
      }
    } else {
      i += 1;
    }
  }
  names
}

struct Stmt {
  text: String,
  names: HashSet<String>,
}

fn statements(src: &str, m: &[u8], b0: usize, b1: usize) -> Vec<Stmt> {
  let mut out = vec![];
  let mut i = b0;
  while i < b1 {
    let boundary = i == 0 || !is_ident(m[i - 1]);
    let (end, names) = if boundary && m[i..].starts_with(b"assert!(") {
      let e = close(m, i + 8, b'(', b')');
      let args = split_top(std::str::from_utf8(&m[i + 8..e - 1]).unwrap(), false);
      (e, idents(&args[0]))
    } else if boundary && CHECKS.iter().any(|c| m[i..].starts_with(c)) {
      let open = m[i..]
        .iter()
        .position(|&c| c == b'(')
        .map_or(b1, |p| i + p + 1);
      let e = close(m, open, b'(', b')');
      (e, idents(std::str::from_utf8(&m[open..e - 1]).unwrap()))
    } else if m[i..].starts_with(b".assert_") {
      let mut k = i + 8;
      while k < b1 && (m[k].is_ascii_lowercase() || m[k] == b'_') {
        k += 1;
      }
      let mut r = i;
      while r > b0 && is_ident(m[r - 1]) {
        r -= 1;
      }
      if m[k..].starts_with(b"()") && r < i {
        (k + 2, HashSet::from([src[r..i].to_string()]))
      } else {
        i += 1;
        continue;
      }
    } else {
      i += 1;
      continue;
    };
    let start = if m[i] == b'.' {
      let mut r = i;
      while r > b0 && is_ident(m[r - 1]) {
        r -= 1;
      }
      r
    } else {
      i
    };
    let mut e = end;
    let mut j = e;
    while j < b1 && (m[j] == b' ' || m[j] == b'\t') {
      j += 1;
    }
    if j < b1 && m[j] == b';' {
      e = j + 1;
    }
    out.push(Stmt {
      text: squash(&src[start..e], &m[start..e]),
      names,
    });
    i = e;
  }
  out
}

struct Fun {
  name: String,
  public: bool,
  params: Vec<(String, String)>,
  body: (usize, usize),
}

struct Impl {
  ty: String,
  funs: Vec<Fun>,
}

fn parse_impls(src: &str, m: &[u8]) -> Vec<Impl> {
  let mut impls = vec![];
  let mut at = 0;
  while let Some(p) = src[at..].find("impl") {
    let i = at + p;
    at = i + 4;
    if (i != 0 && m[i - 1] != b'\n') || m.get(i + 4).is_none_or(|&c| is_ident(c)) {
      continue;
    }
    let mut k = i + 4;
    while m[k] == b' ' {
      k += 1;
    }
    if m[k] == b'<' {
      k = close(m, k + 1, b'<', b'>');
    }
    let Some(brace) = m[k..].iter().position(|&c| c == b'{').map(|p| k + p) else {
      continue;
    };
    let head = std::str::from_utf8(&m[k..brace]).unwrap();
    let head = head.split("where").next().unwrap();
    if head
      .split(|c: char| !(c.is_alphanumeric() || c == '_'))
      .any(|w| w == "for")
    {
      continue;
    }
    let ty: String = head
      .trim()
      .chars()
      .take_while(|&c| c.is_alphanumeric() || c == '_')
      .collect();
    let end = close(m, brace + 1, b'{', b'}');
    let mut funs = vec![];
    let mut depth = 0i32;
    let mut j = brace + 1;
    while j < end - 1 {
      match m[j] {
        b'{' => depth += 1,
        b'}' => depth -= 1,
        b'f'
          if depth == 0
            && m[j..].starts_with(b"fn ")
            && (m[j - 1] == b' ' || m[j - 1] == b'\n') =>
        {
          let mut a = j + 3;
          while m[a] == b' ' {
            a += 1;
          }
          let nm0 = a;
          while is_ident(m[a]) {
            a += 1;
          }
          let name = src[nm0..a].to_string();
          while m[a] == b' ' || m[a] == b'\n' {
            a += 1;
          }
          if m[a] == b'<' {
            a = close(m, a + 1, b'<', b'>');
          }
          while m[a] == b' ' || m[a] == b'\n' {
            a += 1;
          }
          let p1 = close(m, a + 1, b'(', b')');
          let params = split_top(&src[a + 1..p1 - 1], true)
            .iter()
            .filter_map(|p| {
              let p = p.trim();
              let (n, t) = p.split_once(':')?;
              (!p.ends_with("self")).then(|| {
                (
                  n.replace("mut ", "").trim().to_string(),
                  t.chars().filter(|c| !c.is_whitespace()).collect(),
                )
              })
            })
            .collect();
          let bo = m[p1..].iter().position(|&c| c == b'{').map(|p| p1 + p);
          let Some(bo) = bo else {
            j += 1;
            continue;
          };
          let bc = close(m, bo + 1, b'{', b'}');
          let before = src[..j].trim_end();
          let before = before.strip_suffix("const").unwrap_or(before).trim_end();
          let public = before.ends_with("pub")
            || (before.ends_with(')')
              && before
                .rsplit('\n')
                .next()
                .is_some_and(|l| l.contains("pub(")));
          funs.push(Fun {
            name,
            public,
            params,
            body: (bo + 1, bc - 1),
          });
          j = bc;
          continue;
        }
        _ => {}
      }
      j += 1;
    }
    impls.push(Impl { ty, funs });
  }
  impls
}

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
