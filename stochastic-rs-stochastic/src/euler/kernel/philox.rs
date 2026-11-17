//! Philox-4x32-10 and the draws on it in CUDA C, word for word the generator of `euler::philox`.
//! Salmon, Moraes, Dror & Shaw (2011), "Parallel random numbers: as easy as 1, 2, 3", SC'11.

use crate::euler::philox;

/// The counter sites as C constants, written from the Rust table so the two cannot drift.
pub(crate) fn sites() -> String {
  [
    ("site_noise", philox::SITE_NOISE),
    ("site_uniforms", philox::SITE_UNIFORMS),
    ("site_count", philox::SITE_COUNT),
    ("site_sizes", philox::SITE_SIZES),
    ("site_gamma", philox::SITE_GAMMA),
    ("site_series", philox::SITE_SERIES),
    ("site_series_time", philox::SITE_SERIES_TIME),
    ("site_table", philox::SITE_TABLE),
    ("site_series_marks", philox::SITE_SERIES_MARKS),
    ("gamma_boost", philox::GAMMA_BOOST),
  ]
  .map(|(name, value)| format!("#define {name} {value}u\n"))
  .concat()
}

pub(crate) const C_PHILOX: &str = r#"__device__ __forceinline__ void stoch_philox(unsigned int c0, unsigned int c1, unsigned int c2,
    unsigned int c3, unsigned int k0, unsigned int k1, unsigned int* w) {
    for (unsigned int r = 0u; r < 10u; r++) {
        unsigned int hi0 = __umulhi(0xD2511F53u, c0);
        unsigned int lo0 = 0xD2511F53u * c0;
        unsigned int hi1 = __umulhi(0xCD9E8D57u, c2);
        unsigned int lo1 = 0xCD9E8D57u * c2;
        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    w[0] = c0; w[1] = c1; w[2] = c2; w[3] = c3;
}
"#;

/// The `float` draws: a uniform from a word's top 23 bits, an exponential whose second word
/// refines the first's zero cell, a normal from three words.
pub(crate) const C_DRAWS_F32: &str = r#"__device__ __forceinline__ REAL stoch_u01(unsigned int a) {
    return ((REAL)(a >> 9u) + (REAL)0.5) * (REAL)1.1920928955078125e-7;
}
__device__ __forceinline__ REAL stoch_uniform(unsigned int a, unsigned int b) { return stoch_u01(a); }
__device__ __forceinline__ void stoch_step_uniforms(unsigned int path, unsigned int step,
    unsigned int k0, unsigned int k1, unsigned int* words, REAL* u) {
    stoch_philox(path, step, site_uniforms, 0u, k0, k1, words);
    for (unsigned int k = 0u; k < 4u; k++) { u[k] = stoch_u01(words[k]); }
}
// The fast log's error can take -log u below zero near u = 1, where the root would be NaN.
__device__ __forceinline__ REAL stoch_exp1(unsigned int a, unsigned int b) {
    REAL e = ((a >> 9u) == 0u) ? (REAL)15.942385152878742 - STOCH_FAST_LOG(stoch_u01(b)) : -STOCH_FAST_LOG(stoch_u01(a));
    return e > (REAL)0 ? e : (REAL)0;
}
__device__ __forceinline__ REAL stoch_exp1_accurate(unsigned int a, unsigned int b) {
    REAL e = ((a >> 9u) == 0u) ? (REAL)15.942385152878742 - STOCH_LOG(stoch_u01(b)) : -STOCH_LOG(stoch_u01(a));
    return e > (REAL)0 ? e : (REAL)0;
}
__device__ __forceinline__ REAL stoch_normal(unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
    return STOCH_SQRT((REAL)2 * stoch_exp1(a, b)) * STOCH_FAST_COS((REAL)6.283185307179586 * stoch_u01(c));
}
"#;

/// The `double` draws: a uniform and an exponential from 52 bits of two words, a normal from four,
/// and a step's uniform `k` from word `k` of two calls.
pub(crate) const C_DRAWS_F64: &str = r#"__device__ __forceinline__ REAL stoch_uniform(unsigned int a, unsigned int b) {
    return ((REAL)((((U64)a) << 20) | (U64)(b >> 12u)) + (REAL)0.5) * (REAL)2.220446049250313e-16;
}
__device__ __forceinline__ void stoch_step_uniforms(unsigned int path, unsigned int step,
    unsigned int k0, unsigned int k1, unsigned int* words, REAL* u) {
    unsigned int low[4];
    stoch_philox(path, step, site_uniforms, 0u, k0, k1, words);
    stoch_philox(path, step, site_uniforms, 1u, k0, k1, low);
    for (unsigned int k = 0u; k < 4u; k++) { u[k] = stoch_uniform(words[k], low[k]); }
}
__device__ __forceinline__ REAL stoch_exp1(unsigned int a, unsigned int b) {
    REAL e = -STOCH_FAST_LOG(stoch_uniform(a, b));
    return e > (REAL)0 ? e : (REAL)0;
}
__device__ __forceinline__ REAL stoch_exp1_accurate(unsigned int a, unsigned int b) {
    REAL e = -STOCH_LOG(stoch_uniform(a, b));
    return e > (REAL)0 ? e : (REAL)0;
}
__device__ __forceinline__ REAL stoch_normal(unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
    return STOCH_SQRT((REAL)2 * stoch_exp1(a, b)) * STOCH_FAST_COS((REAL)6.283185307179586 * stoch_uniform(c, d));
}
"#;
