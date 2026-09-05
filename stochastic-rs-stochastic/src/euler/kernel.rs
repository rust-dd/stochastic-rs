//! The Euler–Maruyama kernel body the native CUDA and Metal back-ends share:
//! one text, rendered per shading language, with the drift / diffusion blocks
//! spliced in from the family declarations in [`super::families`].
//!
//! The thread index `path`, the output and parameter buffers and the launch
//! arguments (`family`, `components`, `noises`, `x0`, `dt`, `sqrt_dt`,
//! `seed`, `steps`, `paths`, `first_path`, `increments`, `has_curve`,
//! `jump_lambda`, `has_jumps`) and the `incs` and `curve` buffers are bound
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
//! process reaches the same families.
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

/// The per-thread frame around the generated family blocks: the path guard,
/// the counter-hash normals and the write-back. The state, the reported
/// values and the noise are four-slot arrays whatever the family's own
/// arity, so one kernel serves every family. `STEP` and `REPORT` are the
/// blocks [`super::families`] generates from the family declarations, and
/// `REAL`, `STOCH_SQRT`, `STOCH_LOG`, `STOCH_COS`, `STOCH_SIN`, `STOCH_TANH`
/// and the
/// 64-bit buffer index type `INDEX` are the precision placeholders.
pub(crate) const FRAME: &str = r#"    if (path >= paths) return;
    INDEX base = (INDEX)path * steps;
    INDEX plane = (INDEX)paths * steps;
    REAL state[4];
    REAL reported[4];
    REAL noise[4];
    REAL ct = (REAL)0;
    if (has_curve != 0u) { ct = curve[0]; }
    REAL nj = (REAL)0;
    REAL u = (REAL)0;
    REAL u2 = (REAL)0;
    for (unsigned int c = 0u; c < 4u; c++) { state[c] = x0[c]; reported[c] = x0[c]; }
    for (unsigned int c = 0u; c < 4u; c++) { noise[c] = (REAL)0; }
REPORT
    for (unsigned int c = 0u; c < components; c++) { out[(INDEX)c * plane + base] = reported[c]; }
    for (unsigned int i = 1; i < steps; i++) {
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
            noise[0] = incs[(INDEX)path * (steps - 1) + (i - 1)];
        }
        if (has_curve != 0u) { ct = curve[i]; }
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
  /// The type of a buffer index; `unsigned long long` on CUDA, `uint` in MSL.
  pub index: &'a str,
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
    .replace("REAL", lang.real)
}
