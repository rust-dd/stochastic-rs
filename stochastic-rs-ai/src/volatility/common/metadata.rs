use std::collections::HashMap;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;

use super::scaler::StandardScaler;
use super::spec::StochVolModelSpec;

pub(super) const FORMAT_VERSION: usize = 2;

pub(super) fn serialize_metadata(spec: &StochVolModelSpec, scaler: &StandardScaler) -> String {
  let mut out = String::new();
  out.push_str(&format!("version={FORMAT_VERSION}\n"));
  out.push_str(&format!("model_id={}\n", spec.model_id));
  out.push_str(&format!("hidden_dim={}\n", spec.hidden_dim));
  out.push_str(&format!("param_lb={}\n", join_f32(&spec.param_lb)));
  out.push_str(&format!("param_ub={}\n", join_f32(&spec.param_ub)));
  out.push_str(&format!("strikes={}\n", join_f64(&spec.strikes)));
  out.push_str(&format!("maturities={}\n", join_f64(&spec.maturities)));
  out.push_str(&format!("surface_mean={}\n", join_f32(&scaler.mean)));
  out.push_str(&format!("surface_std={}\n", join_f32(&scaler.std)));
  out
}

pub(super) fn parse_metadata(s: &str) -> Result<HashMap<String, String>> {
  let mut out = HashMap::new();
  for line in s.lines() {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
      continue;
    }
    let (k, v) = line
      .split_once('=')
      .ok_or_else(|| anyhow!("invalid metadata line: {line}"))?;
    out.insert(k.trim().to_string(), v.trim().to_string());
  }
  Ok(out)
}

pub(super) fn parse_usize_field(map: &HashMap<String, String>, key: &str) -> Result<usize> {
  let raw = map
    .get(key)
    .ok_or_else(|| anyhow!("missing '{key}' in metadata"))?;
  raw
    .parse::<usize>()
    .with_context(|| format!("failed to parse metadata field '{key}'"))
}

pub(super) fn parse_vec_field(map: &HashMap<String, String>, key: &str) -> Result<Vec<f32>> {
  let raw = map
    .get(key)
    .ok_or_else(|| anyhow!("missing '{key}' in metadata"))?;
  if raw.is_empty() {
    return Ok(Vec::new());
  }
  raw
    .split(',')
    .map(|v| {
      v.parse::<f32>()
        .with_context(|| format!("failed to parse a float in metadata field '{key}'"))
    })
    .collect()
}

pub(super) fn parse_f64_vec_field(map: &HashMap<String, String>, key: &str) -> Result<Vec<f64>> {
  let raw = map
    .get(key)
    .ok_or_else(|| anyhow!("missing '{key}' in metadata"))?;
  if raw.is_empty() {
    return Ok(Vec::new());
  }
  raw
    .split(',')
    .map(|v| {
      v.parse::<f64>()
        .with_context(|| format!("failed to parse a float in metadata field '{key}'"))
    })
    .collect()
}

fn join_f64(values: &[f64]) -> String {
  values
    .iter()
    .map(|v| format!("{v:?}"))
    .collect::<Vec<String>>()
    .join(",")
}

fn join_f32(values: &[f32]) -> String {
  values
    .iter()
    .map(|v| format!("{v:.9}"))
    .collect::<Vec<String>>()
    .join(",")
}
