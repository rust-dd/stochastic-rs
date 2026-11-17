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
    REAL u3 = (REAL)0;
    REAL u4 = (REAL)0;
    unsigned int words[4];
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
        for (unsigned int k = increments; k < NOISES; k++) {
            stoch_philox(first_path + path, i, site_noise + k, 0u, seed_lo, seed_hi, words);
            noise[k] = sqrt_dt * stoch_normal(words[0], words[1], words[2], words[3]);
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

const FRAME_UNIFORMS: &str = r#"        REAL uniforms[4];
        stoch_step_uniforms(first_path + path, i, seed_lo, seed_hi, words, uniforms);
        u = uniforms[0];
        u2 = uniforms[1];
        u3 = uniforms[2];
        u4 = uniforms[3];"#;

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
