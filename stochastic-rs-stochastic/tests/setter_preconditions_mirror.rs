//! Every `with_*` setter runs the check its type's `new` runs on the same argument: the field's one
//! `check_*` or the argument's own `assert_*`, and no assert of its own unless on `self.`.

use std::collections::HashSet;

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

mod setter_preconditions_mirror {
  mod rule;
}
