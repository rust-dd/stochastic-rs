//! The survey's laws and mpmath values (`survey_refs.txt`), one row `stratum key | item=value …`
//! each, `-` past mpmath's reach; read, never regenerated, since `10^x` differs by platform.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::survey::Case;
use crate::survey::Law;
use crate::survey::Stratum;

const TABLE: &str = include_str!("survey_refs.txt");

/// One law's references by item: `mean`, `variance`, `skewness`, `kurtosis`, `entropy`, `cdf@x`, `mgf@t`.
pub struct Row(HashMap<String, Option<f64>>);

impl Row {
  /// `None` where the row names no such item, `Some(None)` where mpmath did not reach it.
  pub fn get(&self, what: &str) -> Option<Option<f64>> {
    self.0.get(what).copied()
  }
}

/// The rows a survey reads, and whether every law it runs must have one.
pub struct Pinned {
  rows: Option<&'static HashMap<String, Row>>,
}

/// The table's laws in its order, each with its stratum and the mgf arguments its row names.
pub fn committed_cases() -> Vec<Case> {
  TABLE
    .lines()
    .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    .map(|line| {
      let (head, items) = line
        .split_once(" | ")
        .expect("a row is `stratum key | items`");
      let (stratum, key) = head.split_once(' ').expect("a row names its stratum");
      let ts = items
        .split_whitespace()
        .filter_map(|item| item.strip_prefix("mgf@")?.split_once('='))
        .map(|(t, _)| t.parse().expect("an mgf argument parses"))
        .collect();
      Case {
        stratum: Stratum::from_name(stratum),
        law: Law::from_key(key),
        ts,
      }
    })
    .collect()
}

impl Pinned {
  pub fn committed() -> Self {
    static ROWS: OnceLock<HashMap<String, Row>> = OnceLock::new();
    Self {
      rows: Some(ROWS.get_or_init(|| parse(TABLE))),
    }
  }

  /// No rows: a run whose values an outside reference checks.
  #[allow(dead_code)]
  pub fn none() -> Self {
    Self { rows: None }
  }

  pub fn required(&self) -> bool {
    self.rows.is_some()
  }

  pub fn row(&self, key: &str) -> Option<&'static Row> {
    self.rows.and_then(|rows| rows.get(key))
  }
}

fn parse(table: &str) -> HashMap<String, Row> {
  let mut rows = HashMap::new();
  for line in table
    .lines()
    .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
  {
    let (head, items) = line
      .split_once(" | ")
      .expect("a row is `stratum key | items`");
    let key = head.split_once(' ').expect("a row names its stratum").1;
    let items = items
      .split_whitespace()
      .map(|item| {
        let (what, value) = item.split_once('=').expect("an item is `what=value`");
        let value = (value != "-").then(|| value.parse::<f64>().expect("a value parses"));
        (what.to_string(), value)
      })
      .collect();
    rows.insert(key.to_string(), Row(items));
  }
  rows
}
