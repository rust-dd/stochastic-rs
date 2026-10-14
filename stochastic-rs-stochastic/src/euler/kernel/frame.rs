//! Ordered CUDA frame fragments; optional blocks follow the launch shape.

mod draws;

mod memory;

use draws::{FRAME_COUNTS, FRAME_JUMP_LAWS};
use memory::{
  FRAME_HISTORY, FRAME_LIFT, FRAME_SERIES_DRAW, FRAME_SERIES_LIVE, FRAME_TABLE_DRAW,
  FRAME_TABLE_LIVE,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Need {
  Always,
  Series,
  Table,
  Program,
  Lift,
  History,
  Counts,
  Jumps,
  Uniforms,
}

pub(super) const FRAME_BLOCKS: [(&str, Need); 14] = [
  (FRAME_LOCALS, Need::Always),
  (FRAME_SERIES_DRAW, Need::Series),
  (FRAME_TABLE_DRAW, Need::Table),
  (FRAME_ENTRY_REPORT, Need::Always),
  (FRAME_NOISE, Need::Always),
  (FRAME_UNIFORMS, Need::Uniforms),
  (FRAME_SERIES_LIVE, Need::Series),
  (FRAME_TABLE_LIVE, Need::Table),
  (FRAME_COUNTS, Need::Counts),
  (FRAME_JUMP_LAWS, Need::Jumps),
  (FRAME_PROGRAM, Need::Program),
  (FRAME_LIFT, Need::Lift),
  (FRAME_HISTORY, Need::History),
  (FRAME_STEP, Need::Always),
];

const FRAME_LOCALS: &str = r#"    if (path >= paths) return;
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
    REAL lv = (REAL)0;
    REAL lift[3];
    lift[0] = (REAL)0; lift[1] = (REAL)0; lift[2] = (REAL)0;
    REAL lh[176];
    REAL lj[176];
    if (has_lift != 0u) {
        for (unsigned int l = 0u; l < lift_n; l++) { lh[l] = (REAL)0; lj[l] = (REAL)0; }
    }
    REAL cv = (REAL)0;
    REAL hist_in[1];
    hist_in[0] = (REAL)0;
    REAL block[512];
    REAL sj = (REAL)0;
    REAL gj = (REAL)0;
    REAL ej = (REAL)0;
    REAL uj = (REAL)0;
    REAL uv = (REAL)0;
    REAL series_size[1];
    series_size[0] = (REAL)0;
    REAL pv = (REAL)0;
    REAL pv2 = (REAL)0;"#;

const FRAME_ENTRY_REPORT: &str = r#"    for (unsigned int c = 0u; c < 4u; c++) { state[c] = x0[c]; reported[c] = x0[c]; }
    for (unsigned int c = 0u; c < 4u; c++) { noise[c] = (REAL)0; }
REPORT
    if (step_first == 0u) {
        for (unsigned int c = 0u; c < COMPONENTS; c++) { out[(INDEX)c * plane + base] = reported[c]; }
    }"#;

const FRAME_NOISE: &str = r#"    for (unsigned int i = (step_first != 0u ? 0u : 1u); i < steps; i++) {
        // Mix path and step separately so their product cannot wrap and repeat noise.
        unsigned int g = (first_path + path) * 2654435761u;
        g ^= (i + 2246822519u) * 2246822519u;
        g ^= g >> 16; g *= 2246822519u; g ^= g >> 13; g *= 3266489917u; g ^= g >> 16;
        for (unsigned int k = 0u; k < NOISES; k++) {
            unsigned int a = (g ^ (2654435769u + k * 2654435761u)) ^ (seed * 2654435761u);
            a ^= a >> 16; a *= 2246822519u; a ^= a >> 13; a *= 3266489917u; a ^= a >> 16;
            unsigned int b = (g ^ (1442695041u + k * 1013904223u)) ^ (seed * 668265263u);
            b ^= b >> 16; b *= 2246822519u; b ^= b >> 13; b *= 3266489917u; b ^= b >> 16;
            REAL u1 = (REAL)a * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6;
            REAL u2 = (REAL)b * (REAL)2.3283064e-10;
            REAL z = STOCH_SQRT((REAL)-2.0 * STOCH_FAST_LOG(u1)) * STOCH_FAST_COS((REAL)6.283185307179586 * u2);
            noise[k] = sqrt_dt * z;
        }
        if (increments != 0u) {
            unsigned int inc_len = steps;
            unsigned int inc_at = i;
            if (step_first == 0u) { inc_len = steps - 1u; inc_at = i - 1u; }
            noise[0] = incs[(INDEX)(path * increments) * inc_len + inc_at];
            if (increments > 1u) {
                noise[1] = incs[(INDEX)(path * increments + 1u) * inc_len + inc_at];
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
"#;

const FRAME_UNIFORMS: &str = r#"        unsigned int hu = (g ^ 2135587861u) ^ (seed * 2654435761u);
        hu ^= hu >> 16; hu *= 2246822519u; hu ^= hu >> 13; hu *= 3266489917u; hu ^= hu >> 16;
        u = (REAL)hu * (REAL)2.3283064e-10;
        unsigned int hv = (g ^ 3266489917u) ^ (seed * 2654435761u);
        hv ^= hv >> 16; hv *= 2246822519u; hv ^= hv >> 13; hv *= 3266489917u; hv ^= hv >> 16;
        u2 = (REAL)hv * (REAL)2.3283064e-10;"#;

const FRAME_PROGRAM: &str = r#"        if (program_n != 0u) {
            for (unsigned int w = 0u; w < program_n; w++) {
                unsigned int plen = (unsigned int)program[w];
                unsigned int pbase = 2u;
                if (w == 1u) { pbase = 2u + 2u * (unsigned int)program[0]; }
                REAL pst[8];
                unsigned int psp = 0u;
                for (unsigned int k = 0u; k < plen; k++) {
                    unsigned int code = (unsigned int)program[pbase + 2u * k];
                    REAL val = program[pbase + 2u * k + 1u];
                    if (code == 0u) { pst[psp] = ct; psp++; }
                    else if (code == 1u) { pst[psp] = state[0]; psp++; }
                    else if (code == 2u) { pst[psp] = val; psp++; }
                    else if (code == 3u) { pst[psp - 2u] = pst[psp - 2u] + pst[psp - 1u]; psp--; }
                    else if (code == 4u) { pst[psp - 2u] = pst[psp - 2u] - pst[psp - 1u]; psp--; }
                    else if (code == 5u) { pst[psp - 2u] = pst[psp - 2u] * pst[psp - 1u]; psp--; }
                    else if (code == 6u) { pst[psp - 2u] = pst[psp - 2u] / pst[psp - 1u]; psp--; }
                    else if (code == 7u) { pst[psp - 2u] = STOCH_POW(pst[psp - 2u], pst[psp - 1u]); psp--; }
                    else if (code == 8u) { pst[psp - 2u] = (pst[psp - 2u] > pst[psp - 1u]) ? pst[psp - 2u] : pst[psp - 1u]; psp--; }
                    else if (code == 9u) { pst[psp - 2u] = (pst[psp - 2u] < pst[psp - 1u]) ? pst[psp - 2u] : pst[psp - 1u]; psp--; }
                    else if (code == 10u) { pst[psp - 1u] = -pst[psp - 1u]; }
                    else if (code == 11u) { pst[psp - 1u] = STOCH_SQRT(pst[psp - 1u]); }
                    else if (code == 12u) { pst[psp - 1u] = STOCH_EXP(pst[psp - 1u]); }
                    else if (code == 13u) { pst[psp - 1u] = STOCH_LOG(pst[psp - 1u]); }
                    else if (code == 14u) { pst[psp - 1u] = STOCH_ABS(pst[psp - 1u]); }
                    else if (code == 15u) { pst[psp - 1u] = STOCH_TANH(pst[psp - 1u]); }
                }
                if (w == 0u) { pv = pst[0]; } else { pv2 = pst[0]; }
            }
        }"#;

const FRAME_STEP: &str = r#"STEP
        if (has_lift != 0u) {
            for (unsigned int l = 0u; l < lift_n; l++) {
                lh[l] = lift_decay[l] * lh[l] + lift_drift_scale[l] * lift[0];
                lj[l] = lift_decay[l] * (lj[l] + lift[1] * lift[2]);
            }
        }
        for (unsigned int c = 0u; c < 4u; c++) { reported[c] = state[c]; }
REPORT
        for (unsigned int c = 0u; c < COMPONENTS; c++) { out[(INDEX)c * plane + base + i] = reported[c]; }
    }
"#;
