//! The Euler–Maruyama kernel body the native CUDA and Metal back-ends share:
//! one text, rendered per shading language. The thread index `path`, the
//! output and parameter buffers and the launch arguments (`family`, `x0`,
//! `dt`, `sqrt_dt`, `seed`, `steps`, `paths`, `first_path`) are bound by the
//! language-specific header around it; the body itself only uses the
//! placeholders [`Language`] fills in. Two decorrelated uniforms per step come
//! from a Murmur3-style integer hash of `(first_path + path, step, seed)`,
//! so a batch produced in chunks is bit-identical to one launch. CIR (family
//! 2) uses full truncation (Lord, Koekkoek & van Dijk 2010).

/// The per-thread body, with `REAL`, `SQRT`, `LOG`, `COS` and the 64-bit
/// buffer index type `INDEX` left as placeholders.
pub(crate) const BODY: &str = r#"    if (path >= paths) return;
    INDEX base = (INDEX)path * steps;
    REAL x = x0;
    REAL reported = x0;
    if (family == 2u && x0 < (REAL)0) reported = (REAL)0;
    out[base] = reported;
    for (unsigned int i = 1; i < steps; i++) {
        unsigned int g = (first_path + path) * steps + i;
        unsigned int a = (g * 2u) ^ (seed * 2654435761u);
        a ^= a >> 16; a *= 2246822519u; a ^= a >> 13; a *= 3266489917u; a ^= a >> 16;
        unsigned int b = (g * 2u + 1u) ^ (seed * 668265263u);
        b ^= b >> 16; b *= 2246822519u; b ^= b >> 13; b *= 3266489917u; b ^= b >> 16;
        REAL u1 = (REAL)a * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6;
        REAL u2 = (REAL)b * (REAL)2.3283064e-10;
        REAL z = SQRT((REAL)-2.0 * LOG(u1)) * COS((REAL)6.283185307179586 * u2);
        if (family == 0u) {
            x = x + params[0] * x * dt + params[1] * x * sqrt_dt * z;
        } else if (family == 1u) {
            x = x + params[0] * (params[1] - x) * dt + params[2] * sqrt_dt * z;
        } else {
            REAL positive = x > (REAL)0 ? x : (REAL)0;
            x = x + params[0] * (params[1] - positive) * dt + params[2] * SQRT(positive) * sqrt_dt * z;
        }
        reported = x;
        if (family == 2u && x < (REAL)0) reported = (REAL)0;
        out[base + i] = reported;
    }
"#;

/// What a shading language substitutes into [`BODY`].
pub(crate) struct Language<'a> {
  /// The scalar type (`float`, `double`).
  pub real: &'a str,
  /// The square-root, natural-log and cosine intrinsics for `real`.
  pub sqrt: &'a str,
  pub log: &'a str,
  pub cos: &'a str,
  /// The type of a buffer index; `unsigned long long` on CUDA, `uint` in MSL.
  pub index: &'a str,
}

/// [`BODY`] with the placeholders of `lang` filled in.
pub(crate) fn render(lang: &Language<'_>) -> String {
  BODY
    .replace("INDEX", lang.index)
    .replace("SQRT", lang.sqrt)
    .replace("LOG", lang.log)
    .replace("COS", lang.cos)
    .replace("REAL", lang.real)
}
