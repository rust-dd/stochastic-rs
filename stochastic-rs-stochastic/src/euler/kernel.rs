//! The Euler–Maruyama kernel body the native CUDA and Metal back-ends share:
//! one text, rendered per shading language, with the drift / diffusion blocks
//! spliced in from the family declarations in [`super::families`].
//!
//! The thread index `path`, the output and parameter buffers and the launch
//! arguments (`family`, `components`, `noises`, `x0`, `dt`, `sqrt_dt`,
//! `seed`, `steps`, `paths`, `first_path`, `increments`, `n_curves`,
//! `jump_lambda`, `has_jumps`, `jump_law`, `jump_a`, `jump_b`, `jump_c`,
//! `step_first`, `gamma_law`, `g1_shape`, `g1_scale`, `g1_per`,
//! `g2_shape`, `g2_scale`, `g2_per`) and
//! the `incs` and `curve` buffers are bound
//! by the language-specific header around the body;
//! the body itself only uses the placeholders [`Language`] fills in. Two
//! decorrelated uniforms per noise component per step come from a
//! Murmur3-style integer hash of `(first_path + path, step, seed)`, so a
//! batch produced in chunks is bit-identical to one launch. The first
//! component hashes that counter directly and every further one hashes it
//! xored with a constant of its own, which leaves a single-noise family's
//! stream exactly what it was before the engine learned about systems. The
//! salt is xored rather than multiplied in because a shading language may
//! constant-fold the multiplication and reject it as an overflow. When
//! `increments` is set the first component reads from `incs` instead — one
//! row of `steps - 1` increments per path — which is how a fractional
//! process reaches the same families. `increments` is a count, not a flag:
//! at two the second component reads the buffer's next `paths` rows, so one
//! embedding feeds a correlated fractional pair without the kernel binding a
//! second buffer. A row is `steps - 1` increments long, one per step the
//! frame takes after the initial state — or `steps` under `step_first`,
//! where the first grid point is a draw and every point consumes one.
//!
//! With `step_first` the frame takes a step before writing the first point,
//! which is what a process whose first grid point is itself a draw needs: a
//! conditional-variance model's series starts at `σ₀ z₀`, not at a
//! deterministic level. Without it the first point is the reported initial
//! state, as every diffusion here wants.
//!
//! A launch writes `components` planes of `paths * steps` values each, so a
//! one-component family fills the buffer exactly as it always did and a
//! system's components come back as separate contiguous paths.
//!
//! A family may read `u` and `u2`, two independent uniforms in `[0, 1)` for
//! the step, from hash streams no noise component and no jump count uses. It is what a scheme with
//! a branch of its own needs — the quadratic-exponential variance step draws
//! it, and the Chambers-Mallows-Stuck stable draw takes both — and a family
//! that never names them pays two integer hashes for them.
//!
//! A family may read `gm` and `gm2`, one or two Gamma draws for the step by
//! Marsaglia-Tsang, with the shape boosted below one exactly as that method
//! prescribes. A draw's shape may carry a term proportional to the step's own
//! jump count, which is what a compound sum of gamma jumps is: the sum of `k`
//! of them is one draw at `k` times the shape. The rejection loop is bounded at 24 tries: the method accepts
//! on the first try better than 98 % of the time for any shape, so exhausting
//! the bound has probability below `1e-40`, and a step that did would take
//! its last candidate rather than loop forever.
//!
//! A family may read `js`, the sum of the step's jump sizes: one normal draw
//! when the sizes are normal, since the sum of `n` of those is itself normal,
//! and a bounded loop when they are double-exponential or tempered-stable,
//! neither of which has such an aggregation. The tempered-stable law thins
//! its own candidates, so the sum is over the accepted ones. It is zero for a family that declares no size law.
//!
//! A family may read `nj`, the number of jumps the step saw: a Poisson draw
//! with mean `jump_lambda · dt`, by Knuth's product of uniforms from a hash
//! stream of its own. It is drawn once per step, so every component of a
//! system sees the same count, and it is zero for a family that declares no
//! intensity.
//!
//! A family may also read `ct`, the step's value of a time-varying
//! coefficient. The host supplies one value per grid point — a short-rate
//! model's `θ(t)`, a term structure of volatilities — and the kernel binds it
//! before each step, so a curve costs one buffer read rather than a
//! parameter per step.
//!
//! A family that needs more than one names them `ct1` through `ct7`: the
//! curves are laid end to end in the same buffer at `steps` values each, and
//! the launch binds only the `n_curves` a family declares, so one curve still
//! costs one read. A dynamic-SABR term structure and a Heath-Jarrow-Morton
//! coefficient set are what the extra slots exist for.

/// The per-thread frame around the generated family blocks: the path guard,
/// the counter-hash normals and the write-back. The state, the reported
/// values and the noise are four-slot arrays whatever the family's own
/// arity, so one kernel serves every family. `STEP` and `REPORT` are the
/// blocks [`super::families`] generates from the family declarations, and
/// `REAL`, `STOCH_SQRT`, `STOCH_LOG`, `STOCH_COS`, `STOCH_SIN`, `STOCH_TANH`,
/// `STOCH_ATAN`
/// and the
/// 64-bit buffer index type `INDEX` are the precision placeholders.
pub(crate) const FRAME: &str = r#"    if (path >= paths) return;
    INDEX base = (INDEX)path * steps;
    INDEX plane = (INDEX)paths * steps;
    REAL state[4];
    REAL reported[4];
    REAL noise[4];
    REAL ct = (REAL)0;
    REAL ct1 = (REAL)0;
    REAL ct2 = (REAL)0;
    REAL ct3 = (REAL)0;
    REAL ct4 = (REAL)0;
    REAL ct5 = (REAL)0;
    REAL ct6 = (REAL)0;
    REAL ct7 = (REAL)0;
    if (n_curves > 0u) { ct = curve[(INDEX)0 * steps]; }
    if (n_curves > 1u) { ct1 = curve[(INDEX)1 * steps]; }
    if (n_curves > 2u) { ct2 = curve[(INDEX)2 * steps]; }
    if (n_curves > 3u) { ct3 = curve[(INDEX)3 * steps]; }
    if (n_curves > 4u) { ct4 = curve[(INDEX)4 * steps]; }
    if (n_curves > 5u) { ct5 = curve[(INDEX)5 * steps]; }
    if (n_curves > 6u) { ct6 = curve[(INDEX)6 * steps]; }
    if (n_curves > 7u) { ct7 = curve[(INDEX)7 * steps]; }
    REAL nj = (REAL)0;
    REAL js = (REAL)0;
    REAL gm = (REAL)0;
    REAL gm2 = (REAL)0;
    REAL u = (REAL)0;
    REAL u2 = (REAL)0;
    for (unsigned int c = 0u; c < 4u; c++) { state[c] = x0[c]; reported[c] = x0[c]; }
    for (unsigned int c = 0u; c < 4u; c++) { noise[c] = (REAL)0; }
REPORT
    if (step_first == 0u) {
        for (unsigned int c = 0u; c < components; c++) { out[(INDEX)c * plane + base] = reported[c]; }
    }
    for (unsigned int i = (step_first != 0u ? 0u : 1u); i < steps; i++) {
        unsigned int g = (first_path + path) * steps + i;
        for (unsigned int k = 0u; k < noises; k++) {
            unsigned int gk = g;
            if (k == 1u) { gk = g ^ 2654435769u; }
            if (k == 2u) { gk = g ^ 2246822519u; }
            if (k == 3u) { gk = g ^ 3266489917u; }
            unsigned int a = (gk * 2u) ^ (seed * 2654435761u);
            a ^= a >> 16; a *= 2246822519u; a ^= a >> 13; a *= 3266489917u; a ^= a >> 16;
            unsigned int b = (gk * 2u + 1u) ^ (seed * 668265263u);
            b ^= b >> 16; b *= 2246822519u; b ^= b >> 13; b *= 3266489917u; b ^= b >> 16;
            REAL u1 = (REAL)a * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6;
            REAL u2 = (REAL)b * (REAL)2.3283064e-10;
            REAL z = STOCH_SQRT((REAL)-2.0 * STOCH_LOG(u1)) * STOCH_COS((REAL)6.283185307179586 * u2);
            noise[k] = sqrt_dt * z;
        }
        if (increments != 0u) {
            unsigned int inc_len = steps;
            unsigned int inc_at = i;
            if (step_first == 0u) { inc_len = steps - 1u; inc_at = i - 1u; }
            noise[0] = incs[(INDEX)path * inc_len + inc_at];
            if (increments > 1u) {
                noise[1] = incs[((INDEX)paths + (INDEX)path) * inc_len + inc_at];
            }
        }
        if (n_curves > 0u) { ct = curve[(INDEX)0 * steps + i]; }
        if (n_curves > 1u) { ct1 = curve[(INDEX)1 * steps + i]; }
        if (n_curves > 2u) { ct2 = curve[(INDEX)2 * steps + i]; }
        if (n_curves > 3u) { ct3 = curve[(INDEX)3 * steps + i]; }
        if (n_curves > 4u) { ct4 = curve[(INDEX)4 * steps + i]; }
        if (n_curves > 5u) { ct5 = curve[(INDEX)5 * steps + i]; }
        if (n_curves > 6u) { ct6 = curve[(INDEX)6 * steps + i]; }
        if (n_curves > 7u) { ct7 = curve[(INDEX)7 * steps + i]; }
        unsigned int hu = (g ^ 2135587861u) ^ (seed * 2654435761u);
        hu ^= hu >> 16; hu *= 2246822519u; hu ^= hu >> 13; hu *= 3266489917u; hu ^= hu >> 16;
        u = (REAL)hu * (REAL)2.3283064e-10;
        unsigned int hv = (g ^ 3266489917u) ^ (seed * 2654435761u);
        hv ^= hv >> 16; hv *= 2246822519u; hv ^= hv >> 13; hv *= 3266489917u; hv ^= hv >> 16;
        u2 = (REAL)hv * (REAL)2.3283064e-10;
        if (has_jumps != 0u) {
            REAL ell = STOCH_EXP(-jump_lambda * dt);
            REAL prod = (REAL)1;
            unsigned int cnt = 0u;
            for (unsigned int j = 0u; j < 64u; j++) {
                unsigned int h = (g ^ (2166136261u + j * 16777619u)) ^ (seed * 374761393u);
                h ^= h >> 16; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
                prod = prod * ((REAL)h * (REAL)2.3283064e-10);
                if (prod <= ell) { break; }
                cnt++;
            }
            nj = (REAL)cnt;
        }
        for (unsigned int gi = 0u; gi < gamma_law; gi++) {
            REAL gsh = ((gi == 0u) ? g1_shape : g2_shape) + ((gi == 0u) ? g1_per : g2_per) * nj;
            REAL gsc = (gi == 0u) ? g1_scale : g2_scale;
            REAL draw = (REAL)0;
            if (gsh > (REAL)0) {
                REAL boost = (REAL)1;
                REAL a = gsh;
                if (a < (REAL)1) {
                    unsigned int hb = (g ^ (2246822519u + gi * 97u)) ^ (seed * 2654435761u);
                    hb ^= hb >> 16; hb *= 2246822519u; hb ^= hb >> 13; hb *= 3266489917u; hb ^= hb >> 16;
                    REAL ub = (REAL)hb * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6;
                    boost = STOCH_POW(ub, (REAL)1 / gsh);
                    a = a + (REAL)1;
                }
                REAL dd = a - (REAL)1 / (REAL)3;
                REAL cc = (REAL)1 / STOCH_SQRT((REAL)9 * dd);
                REAL val = dd;
                for (unsigned int j = 0u; j < 24u; j++) {
                    unsigned int p1 = (g ^ (1103515245u + gi * 7919u + j * 104729u)) ^ (seed * 2654435761u);
                    p1 ^= p1 >> 16; p1 *= 2246822519u; p1 ^= p1 >> 13; p1 *= 3266489917u; p1 ^= p1 >> 16;
                    unsigned int p2 = (g ^ (1013904223u + gi * 7919u + j * 104729u)) ^ (seed * 2654435761u);
                    p2 ^= p2 >> 16; p2 *= 2246822519u; p2 ^= p2 >> 13; p2 *= 3266489917u; p2 ^= p2 >> 16;
                    REAL ga = (REAL)p1 * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6;
                    REAL gb = (REAL)p2 * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6;
                    REAL zg = STOCH_SQRT((REAL)-2.0 * STOCH_LOG(ga)) * STOCH_COS((REAL)6.283185307179586 * gb);
                    REAL vv = (REAL)1 + cc * zg;
                    vv = vv * vv * vv;
                    if (vv > (REAL)0) {
                        unsigned int p3 = (g ^ (668265263u + gi * 7919u + j * 104729u)) ^ (seed * 2654435761u);
                        p3 ^= p3 >> 16; p3 *= 2246822519u; p3 ^= p3 >> 13; p3 *= 3266489917u; p3 ^= p3 >> 16;
                        REAL ug = (REAL)p3 * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6;
                        val = dd * vv;
                        if (STOCH_LOG(ug) < (REAL)0.5 * zg * zg + dd - dd * vv + dd * STOCH_LOG(vv)) { break; }
                    }
                }
                draw = gsc * boost * val;
            }
            if (gi == 0u) { gm = draw; } else { gm2 = draw; }
        }
        js = (REAL)0;
        if (jump_law == 1u) {
            unsigned int ja = (g ^ 1103515245u) ^ (seed * 2654435761u);
            ja ^= ja >> 16; ja *= 2246822519u; ja ^= ja >> 13; ja *= 3266489917u; ja ^= ja >> 16;
            unsigned int jb = (g ^ 1013904223u) ^ (seed * 2654435761u);
            jb ^= jb >> 16; jb *= 2246822519u; jb ^= jb >> 13; jb *= 3266489917u; jb ^= jb >> 16;
            REAL ua = (REAL)ja * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6;
            REAL ub = (REAL)jb * (REAL)2.3283064e-10;
            REAL zj = STOCH_SQRT((REAL)-2.0 * STOCH_LOG(ua)) * STOCH_COS((REAL)6.283185307179586 * ub);
            js = jump_a * nj + jump_b * STOCH_SQRT(nj) * zj;
        }
        if (jump_law == 2u) {
            for (unsigned int j = 0u; j < 32u; j++) {
                if ((REAL)j >= nj) { break; }
                unsigned int ka = (g ^ (2654435761u + j * 40503u)) ^ (seed * 2654435761u);
                ka ^= ka >> 16; ka *= 2246822519u; ka ^= ka >> 13; ka *= 3266489917u; ka ^= ka >> 16;
                unsigned int kb = (g ^ (668265263u + j * 40503u)) ^ (seed * 2654435761u);
                kb ^= kb >> 16; kb *= 2246822519u; kb ^= kb >> 13; kb *= 3266489917u; kb ^= kb >> 16;
                REAL up = (REAL)ka * (REAL)2.3283064e-10;
                REAL ue = (REAL)kb * (REAL)2.3283064e-10;
                REAL ee = -STOCH_LOG((REAL)1 - ue);
                js += (up < jump_a) ? (ee / jump_b) : (-(ee / jump_c));
            }
        }
        if (jump_law == 3u) {
            for (unsigned int j = 0u; j < 32u; j++) {
                if ((REAL)j >= nj) { break; }
                unsigned int ta = (g ^ (2654435761u + j * 40503u)) ^ (seed * 2654435761u);
                ta ^= ta >> 16; ta *= 2246822519u; ta ^= ta >> 13; ta *= 3266489917u; ta ^= ta >> 16;
                unsigned int tb = (g ^ (668265263u + j * 40503u)) ^ (seed * 2654435761u);
                tb ^= tb >> 16; tb *= 2246822519u; tb ^= tb >> 13; tb *= 3266489917u; tb ^= tb >> 16;
                REAL uu1 = (REAL)ta * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6;
                REAL uu2 = (REAL)tb * (REAL)2.3283064e-10;
                REAL xj = jump_a * STOCH_POW(uu1, jump_b);
                if (uu2 <= STOCH_EXP(-jump_c * xj)) { js += xj; }
            }
        }
STEP
        for (unsigned int c = 0u; c < 4u; c++) { reported[c] = state[c]; }
REPORT
        for (unsigned int c = 0u; c < components; c++) { out[(INDEX)c * plane + base + i] = reported[c]; }
    }
"#;

/// What a shading language substitutes into the kernel text.
pub(crate) struct Language<'a> {
  /// The scalar type (`float`, `double`).
  pub real: &'a str,
  /// The square-root, natural-log, cosine, exponential and power intrinsics
  /// for `real`.
  pub sqrt: &'a str,
  pub log: &'a str,
  pub cos: &'a str,
  pub sin: &'a str,
  pub exp: &'a str,
  pub pow: &'a str,
  pub abs: &'a str,
  pub tanh: &'a str,
  pub atan: &'a str,
  /// The type of a buffer index; `unsigned long long` on CUDA, `uint` in MSL.
  pub index: &'a str,
}

/// Metal Shading Language: `f32` only, and a 32-bit buffer index.
///
/// Compiled without the `metal` feature as well, for the same reason
/// [`cuda_language`] is: the rendering checks below cover every table on any
/// machine.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) fn metal_language() -> Language<'static> {
  Language {
    real: "float",
    sqrt: "sqrt",
    log: "log",
    cos: "cos",
    sin: "sin",
    exp: "exp",
    pow: "pow",
    abs: "abs",
    tanh: "tanh",
    atan: "atan",
    index: "uint",
  }
}

/// CUDA C at the precision `real` names. Compiled without the `cuda` feature
/// too, so the rendering checks below cover the CUDA tables on a machine that
/// has no CUDA — which is the only place a missing intrinsic there would
/// otherwise show up.
/// CUDA C at the precision `real` names. The single-precision intrinsics
/// carry the `f` suffix CUDA gives them; a double-precision kernel takes the
/// unsuffixed ones. Both index with 64 bits, since a batch's plane can pass
/// `u32::MAX` elements.
///
/// Compiled without the `cuda` feature as well, so the rendering checks below
/// cover the CUDA tables on a machine that has no CUDA — the only place a
/// missing intrinsic there would otherwise surface is the driver's compiler.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn cuda_language(real: &'static str) -> Language<'static> {
  if real == "float" {
    Language {
      real,
      sqrt: "sqrtf",
      log: "logf",
      cos: "cosf",
      sin: "sinf",
      exp: "expf",
      pow: "powf",
      abs: "fabsf",
      tanh: "tanhf",
      atan: "atanf",
      index: "unsigned long long",
    }
  } else {
    Language {
      real,
      sqrt: "sqrt",
      log: "log",
      cos: "cos",
      sin: "sin",
      exp: "exp",
      pow: "pow",
      abs: "fabs",
      tanh: "tanh",
      atan: "atan",
      index: "unsigned long long",
    }
  }
}

/// The function-vocabulary defines a generated family block may use, with the
/// intrinsics of `lang` filled in. Emitted above the kernel.
pub(crate) fn prelude(lang: &Language<'_>) -> String {
  substitute(super::families::C_PRELUDE, lang)
}

/// The kernel body: the frame with the generated family blocks spliced in and
/// the placeholders of `lang` filled in.
pub(crate) fn render(lang: &Language<'_>) -> String {
  let body = FRAME
    .replace("STEP", super::families::C_STEP.trim_end_matches('\n'))
    .replace("REPORT", super::families::C_REPORT.trim_end_matches('\n'));
  substitute(&body, lang)
}

fn substitute(text: &str, lang: &Language<'_>) -> String {
  text
    .replace("INDEX", lang.index)
    .replace("STOCH_SQRT", lang.sqrt)
    .replace("STOCH_LOG", lang.log)
    .replace("STOCH_COS", lang.cos)
    .replace("STOCH_SIN", lang.sin)
    .replace("STOCH_EXP", lang.exp)
    .replace("STOCH_POW", lang.pow)
    .replace("STOCH_ABS", lang.abs)
    .replace("STOCH_TANH", lang.tanh)
    .replace("STOCH_ATAN", lang.atan)
    .replace("REAL", lang.real)
}

#[cfg(test)]
mod tests {
  use super::cuda_language;
  use super::metal_language;
  use super::prelude;
  use super::render;

  /// Every shading language the engine renders, with a name for the failure
  /// message. `f32` and `f64` CUDA are separate tables, so both are checked.
  fn languages() -> Vec<(&'static str, super::Language<'static>)> {
    vec![
      ("MSL", metal_language()),
      ("CUDA f32", cuda_language("float")),
      ("CUDA f64", cuda_language("double")),
    ]
  }

  /// A placeholder that survives rendering is an intrinsic the vocabulary
  /// gained without a name in every `Language` table. The kernel still
  /// compiles on the host and still passes every test that does not launch
  /// it, and then fails inside the driver's compiler on a machine that has
  /// the device — which is exactly the failure a Mac cannot see for CUDA.
  #[test]
  fn every_placeholder_is_substituted() {
    for (name, lang) in languages() {
      let source = format!("{}{}", prelude(&lang), render(&lang));
      assert!(
        !source.contains("STOCH_"),
        "{name}: an intrinsic placeholder survived rendering — \
         the vocabulary names it but this Language table does not"
      );
      assert!(
        !source.contains("INDEX"),
        "{name}: the buffer-index placeholder survived rendering"
      );
    }
  }

  /// Each language names a real type and an index type, and every intrinsic.
  /// An empty entry renders as a syntax error the device only reports at
  /// launch.
  #[test]
  fn every_intrinsic_is_named() {
    for (name, lang) in languages() {
      for (what, value) in [
        ("real", lang.real),
        ("sqrt", lang.sqrt),
        ("log", lang.log),
        ("cos", lang.cos),
        ("sin", lang.sin),
        ("exp", lang.exp),
        ("pow", lang.pow),
        ("abs", lang.abs),
        ("tanh", lang.tanh),
        ("atan", lang.atan),
        ("index", lang.index),
      ] {
        assert!(!value.is_empty(), "{name}: `{what}` has no intrinsic");
      }
    }
  }

  /// Every declared family reaches the rendered kernel: the step dispatch and
  /// the report dispatch each carry one arm per family code, so a family
  /// declared without being spliced would launch as a no-op.
  #[test]
  fn every_family_reaches_the_rendered_kernel() {
    let lang = metal_language();
    let body = render(&lang);
    for family in super::super::families::Family::ALL {
      let code = family.code();
      assert!(
        body.contains(&format!("family == {code}u")),
        "{family:?} (code {code}) is declared but the rendered kernel never \
         dispatches to it"
      );
    }
  }
}
