//! Expands shared family clauses for host evaluation and CUDA rendering.

macro_rules! euler_family_exprs {
  (@bare_lift [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],) => { "" };

  (@bare_lift [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],
    { drift ($($ld:tt)*) diffusion ($($lg:tt)*) shock ($($lsh:tt)*) }) => {
    concat!(
      euler_family_exprs!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_family_exprs!(@bind_slots "state" [0 1 2 3] $($state)*),
      euler_family_exprs!(@bind_slots "noise" [0 1 2 3] $($noise)*),
      euler_family_exprs!(@c_body "lift", [0 1 2], [lf lg lsh], $($ld)*, $($lg)*, $($lsh)*),
    )
  };

  (@bare_history [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],) => { "" };

  (@bare_history [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],
    { push ($($hp:tt)*) weights ($w:ident) }) => {
    concat!(
      euler_family_exprs!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_family_exprs!(@bind_slots "state" [0 1 2 3] $($state)*),
      euler_family_exprs!(@bind_slots "noise" [0 1 2 3] $($noise)*),
      euler_family_exprs!(@c_body "hist_in", [0], [hp], $($hp)*),
    )
  };

  (@bare_series [$($param:ident)*], [$($state:ident)*],) => { "" };

  (@bare_series [$($param:ident)*], [$($state:ident)*], { size ($($sz:tt)*) }) => {
    euler_family_exprs!(@bare_series_body [$($param)*], [$($state)*], $($sz)*)
  };

  (@bare_series [$($param:ident)*], [$($state:ident)*], { live size ($($sz:tt)*) }) => {
    euler_family_exprs!(@bare_series_body [$($param)*], [$($state)*], $($sz)*)
  };

  (@bare_series_body [$($param:ident)*], [$($state:ident)*], $($sz:tt)*) => {
    concat!(
      euler_family_exprs!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_family_exprs!(@bind_slots "state" [0 1 2 3] $($state)*),
      euler_family_exprs!(@c_body "series_size", [0], [sz], $($sz)*),
    )
  };

  (@bare_table [$($param:ident)*],) => { "" };

  (@bare_table [$($param:ident)*], { increment ($($ti:tt)*) }) => {
    concat!(
      euler_family_exprs!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_family_exprs!(@c_body "table_inc", [0], [ti], $($ti)*),
    )
  };

  (@c_lift $code:literal, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],) => { "" };

  (@c_lift $code:literal, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],
    { drift ($($ld:tt)*) diffusion ($($lg:tt)*) shock ($($lsh:tt)*) }) => {
    concat!(
      "        if (family == ", stringify!($code), "u) {\n",
      euler_family_exprs!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_family_exprs!(@bind_slots "state" [0 1 2 3] $($state)*),
      euler_family_exprs!(@bind_slots "noise" [0 1 2 3] $($noise)*),
      euler_family_exprs!(@c_body "lift", [0 1 2], [lf lg lsh], $($ld)*, $($lg)*, $($lsh)*),
      "        }\n",
    )
  };

  (@host_lift $params:ident, $dt:ident, $state_in:ident, $noise_in:ident, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],) => {
    [T::zero(); 3]
  };

  (@host_lift $params:ident, $dt:ident, $state_in:ident, $noise_in:ident, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],
    { drift ($($ld:tt)*) diffusion ($($lg:tt)*) shock ($($lsh:tt)*) }) => {{
    #[allow(unused_mut, unused_variables)]
    let mut slot = 0;
    $(
      let $param = $params[slot];
      slot += 1;
    )*
    let _ = slot;
    #[allow(unused_mut, unused_variables)]
    let mut at = 0;
    $(
      let $state = $state_in[at];
      at += 1;
    )*
    let _ = at;
    #[allow(unused_mut, unused_variables)]
    let mut at = 0;
    $(
      let $noise = $noise_in[at];
      at += 1;
    )*
    let _ = at;
    let mut lift = [T::zero(); 3];
    euler_family_exprs!(@host_assign lift, $($ld)*, $($lg)*, $($lsh)*);
    lift
  }};

  (@count_exprs bind $n:ident = $e:expr; $($rest:tt)*) => {
    euler_family_exprs!(@count_exprs $($rest)*)
  };

  (@count_exprs $($e:expr),* $(,)?) => { [$(stringify!($e)),*].len() };

  (@history_slot) => { None };

  (@history_slot { push ($($hp:tt)*) weights (ct) }) => { Some(0u32) };

  (@history_slot { push ($($hp:tt)*) weights (ct1) }) => { Some(1u32) };

  (@history_slot { push ($($hp:tt)*) weights (ct2) }) => { Some(2u32) };

  (@history_slot { push ($($hp:tt)*) weights (ct3) }) => { Some(3u32) };

  (@history_slot { push ($($hp:tt)*) weights (ct4) }) => { Some(4u32) };

  (@history_slot { push ($($hp:tt)*) weights (ct5) }) => { Some(5u32) };

  (@history_slot { push ($($hp:tt)*) weights (ct6) }) => { Some(6u32) };

  (@history_slot { push ($($hp:tt)*) weights (ct7) }) => { Some(7u32) };

  (@c_history $code:literal, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],) => { "" };

  (@c_history $code:literal, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],
    { push ($($hp:tt)*) weights ($w:ident) }) => {
    concat!(
      "        if (family == ", stringify!($code), "u) {\n",
      euler_family_exprs!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_family_exprs!(@bind_slots "state" [0 1 2 3] $($state)*),
      euler_family_exprs!(@bind_slots "noise" [0 1 2 3] $($noise)*),
      euler_family_exprs!(@c_body "hist_in", [0], [hp], $($hp)*),
      "        }\n",
    )
  };

  (@host_history $params:ident, $dt:ident, $u:ident, $u2:ident, $state_in:ident, $noise_in:ident, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],) => {
    None
  };

  (@host_history $params:ident, $dt:ident, $u:ident, $u2:ident, $state_in:ident, $noise_in:ident, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],
    { push ($($hp:tt)*) weights ($w:ident) }) => {{
    #[allow(unused_mut, unused_variables)]
    let mut slot = 0;
    $(
      let $param = $params[slot];
      slot += 1;
    )*
    let _ = slot;
    #[allow(unused_mut, unused_variables)]
    let mut at = 0;
    $(
      let $state = $state_in[at];
      at += 1;
    )*
    let _ = at;
    #[allow(unused_mut, unused_variables)]
    let mut at = 0;
    $(
      let $noise = $noise_in[at];
      at += 1;
    )*
    let _ = at;
    let mut pushed = [T::zero(); 1];
    euler_family_exprs!(@host_assign pushed, $($hp)*);
    Some(pushed[0])
  }};

  (@has_series) => { false };

  (@has_series { size ($($sz:tt)*) }) => { true };

  (@has_series { live size ($($sz:tt)*) }) => { true };

  (@series_live) => { false };

  (@series_live { size ($($sz:tt)*) }) => { false };

  (@series_live { live size ($($sz:tt)*) }) => { true };

  (@c_series $code:literal, [$($param:ident)*], [$($state:ident)*],) => { "" };

  (@c_series $code:literal, [$($param:ident)*], [$($state:ident)*], { size ($($sz:tt)*) }) => {
    euler_family_exprs!(@c_series_body $code, [$($param)*], [$($state)*], $($sz)*)
  };

  (@c_series $code:literal, [$($param:ident)*], [$($state:ident)*], { live size ($($sz:tt)*) }) => {
    euler_family_exprs!(@c_series_body $code, [$($param)*], [$($state)*], $($sz)*)
  };

  (@c_series_body $code:literal, [$($param:ident)*], [$($state:ident)*], $($sz:tt)*) => {
    concat!(
      "            if (family == ", stringify!($code), "u) {\n",
      euler_family_exprs!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_family_exprs!(@bind_slots "state" [0 1 2 3] $($state)*),
      euler_family_exprs!(@c_body "series_size", [0], [sz], $($sz)*),
      "            }\n",
    )
  };

  (@host_series $params:ident, $dt:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $state_in:ident, [$($param:ident)*], [$($state:ident)*],) => {
    None
  };

  (@host_series $params:ident, $dt:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $state_in:ident, [$($param:ident)*], [$($state:ident)*],
    { size ($($sz:tt)*) }) => {
    euler_family_exprs!(@host_series_body $params, $state_in, [$($param)*], [$($state)*], $($sz)*)
  };

  (@host_series $params:ident, $dt:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $state_in:ident, [$($param:ident)*], [$($state:ident)*],
    { live size ($($sz:tt)*) }) => {
    euler_family_exprs!(@host_series_body $params, $state_in, [$($param)*], [$($state)*], $($sz)*)
  };

  (@host_series_body $params:ident, $state_in:ident, [$($param:ident)*], [$($state:ident)*], $($sz:tt)*) => {{
    #[allow(unused_mut, unused_variables)]
    let mut slot = 0;
    $(
      let $param = $params[slot];
      slot += 1;
    )*
    let _ = slot;
    #[allow(unused_mut, unused_variables)]
    let mut at = 0;
    $(
      let $state = $state_in[at];
      at += 1;
    )*
    let _ = at;
    let mut size = [T::zero(); 1];
    euler_family_exprs!(@host_assign size, $($sz)*);
    Some(size[0])
  }};

  (@has_table) => { false };

  (@has_table { increment ($($ti:tt)*) }) => { true };

  (@c_table $code:literal, [$($param:ident)*],) => { "" };

  (@c_table $code:literal, [$($param:ident)*], { increment ($($ti:tt)*) }) => {
    concat!(
      "                if (family == ", stringify!($code), "u) {\n",
      euler_family_exprs!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_family_exprs!(@c_body "table_inc", [0], [ti], $($ti)*),
      "                }\n",
    )
  };

  (@host_table $params:ident, $dt:ident, $uj:ident, $uv:ident, $tv:ident, [$($param:ident)*],) => {
    None
  };

  (@host_table $params:ident, $dt:ident, $uj:ident, $uv:ident, $tv:ident, [$($param:ident)*],
    { increment ($($ti:tt)*) }) => {{
    #[allow(unused_mut, unused_variables)]
    let mut slot = 0;
    $(
      let $param = $params[slot];
      slot += 1;
    )*
    let _ = slot;
    let mut inc = [T::zero(); 1];
    euler_family_exprs!(@host_assign inc, $($ti)*);
    Some(inc[0])
  }};

  (@host_assign $out:ident, bind $n:ident = $e:expr; $($rest:tt)*) => {{
    let $n = $e;
    euler_family_exprs!(@host_assign $out, $($rest)*)
  }};

  (@host_assign $out:ident, $($e:expr),* $(,)?) => {{
    let mut at = 0;
    $(
      $out[at] = $e;
      at += 1;
    )*
    let _ = at;
  }};

  (@bind_params [$($idx:literal)*]) => { "" };

  (@bind_params [$i:literal $($rest_idx:literal)*] $head:ident $($rest:ident)*) => {
    concat!(
      "            const REAL ", stringify!($head), " = params[", stringify!($i), "];\n",
      euler_family_exprs!(@bind_params [$($rest_idx)*] $($rest)*)
    )
  };

  (@bind_slots $buf:literal [$($idx:literal)*]) => { "" };

  (@bind_slots $buf:literal [$i:literal $($rest_idx:literal)*] $head:ident $($rest:ident)*) => {
    concat!(
      "            const REAL ", stringify!($head), " = ", $buf, "[", stringify!($i), "];\n",
      euler_family_exprs!(@bind_slots $buf [$($rest_idx)*] $($rest)*)
    )
  };

  (@c_body $lhs:literal, [$($idx:literal)*], [$($rem:ident)*], bind $n:ident = $e:expr; $($rest:tt)*) => {
    concat!(
      "            const REAL ", stringify!($n), " = ", stringify!($e), ";\n",
      euler_family_exprs!(@c_body $lhs, [$($idx)*], [$($rem)*], $($rest)*)
    )
  };

  (@c_body $lhs:literal, [$($idx:literal)*], [$($rem:ident)*], $($e:expr),* $(,)?) => {
    concat!(
      euler_family_exprs!(@c_temps [$($idx)*] $($e),*),
      euler_family_exprs!(@c_store $lhs, [$($idx)*] $($e),*)
    )
  };

  (@c_temps [$($idx:literal)*]) => { "" };

  (@c_temps [$i:literal $($rest_idx:literal)*] $head:expr $(, $rest:expr)*) => {
    concat!(
      "            const REAL __n", stringify!($i), " = ", stringify!($head), ";\n",
      euler_family_exprs!(@c_temps [$($rest_idx)*] $($rest),*)
    )
  };

  (@c_store $lhs:literal, [$($idx:literal)*]) => { "" };

  // One store per expression, not per declared name: a family may keep more
  // state slots than it reports, and the temporaries exist per expression.
  (@c_store $lhs:literal, [$i:literal $($rest_idx:literal)*] $head:expr $(, $rest:expr)*) => {
    concat!(
      "            ", $lhs, "[", stringify!($i), "] = __n", stringify!($i), ";\n",
      euler_family_exprs!(@c_store $lhs, [$($rest_idx)*] $($rest),*)
    )
  };
}
