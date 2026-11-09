//! Authoritative family expressions shared by host, CUDA and Metal Rust.

// The input identifiers are forwarded through each chunk so their bindings
// and uses retain one macro hygiene context after the declarations are joined.

#[macro_use]
mod multifactor;
#[macro_use]
mod memory;
#[macro_use]
mod lifts;
#[macro_use]
mod transformations;
#[macro_use]
mod jump_schemes;
#[macro_use]
mod systems;
#[macro_use]
mod diffusions;

/// Whether stringified clause tokens use `name` as a whole identifier.
pub(crate) const fn reads(source: &str, name: &str) -> bool {
  const fn identifier(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
  }
  let source = source.as_bytes();
  let name = name.as_bytes();
  let mut start = 0;
  while start + name.len() <= source.len() {
    let mut matched = 0;
    while matched < name.len() && source[start + matched] == name[matched] {
      matched += 1;
    }
    if matched == name.len()
      && (start == 0 || !identifier(source[start - 1]))
      && (start + matched == source.len() || !identifier(source[start + matched]))
    {
      return true;
    }
    start += 1;
  }
  false
}

macro_rules! euler_family_definitions {
  ($emit:ident) => {
    euler_diffusions! {
      $emit;
      (params, dt, ct, ct1, ct2, ct3, ct4, ct5, ct6, ct7, nj, js, gm, gm2, u, u2, lv, cv, sj, gj, ej, uj, uv, iv, tv, pv, pv2);
      step_inputs(
      params, dt, ct, ct1, ct2, ct3, ct4, ct5, ct6, ct7, nj, js, gm, gm2, u, u2, lv, cv, sj, gj, ej, uj, uv, iv, tv, pv, pv2,
      state(slot_a, slot_b, slot_c, slot_d),
      noise(shock_a, shock_b, shock_c, shock_d),
      select(component, produced)
      );
    }
  };
}
