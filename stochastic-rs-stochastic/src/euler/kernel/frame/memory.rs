//! CUDA series, clock tables, lifts and history over private per-path storage.

pub(super) const FRAME_SERIES_DRAW: &str = r#"    if (series_n != 0u) {
        for (unsigned int c = 0u; c < 4u; c++) { state[c] = x0[c]; }
        for (unsigned int k = 0u; k < 512u; k++) { block[k] = (REAL)0; }
        for (unsigned int j = 1u; j <= series_n; j++) {
            stoch_philox(first_path + path, 0u, site_series, j, seed_lo, seed_hi, words);
            gj += stoch_exp1(words[0], words[1]);
            if (series_live != 0u) {
                block[j - 1u] = gj;
            } else {
                ej = stoch_exp1(words[2], words[3]);
                stoch_philox(first_path + path, 0u, site_series_marks, j, seed_lo, seed_hi, words);
                uj = stoch_uniform(words[0], words[1]);
                uv = stoch_uniform(words[2], words[3]);
                stoch_philox(first_path + path, 0u, site_series_time, j, seed_lo, seed_hi, words);
                REAL ratio = stoch_uniform(words[0], words[1]) * (REAL)(steps - 1u);
SERIES
                unsigned int cell = (unsigned int)ratio;
                if ((REAL)cell < ratio) { cell += 1u; }
                if (cell < 1u) { cell = 1u; }
                if (cell > steps - 1u) { cell = steps - 1u; }
                block[cell] += series_size[0];
            }
        }
        gj = (REAL)0; ej = (REAL)0; uj = (REAL)0; uv = (REAL)0;
    }"#;

pub(super) const FRAME_TABLE_DRAW: &str = r#"    REAL iv = (REAL)0;
    REAL tv = (REAL)0;
    REAL table_inc[1];
    table_inc[0] = (REAL)0;
    unsigned int tp = 1u;
    if (table_n != 0u) {
        REAL horizon = dt * (REAL)(steps - 1u);
        REAL umax = table_u0;
        if (umax <= (REAL)0) { umax = (horizon > (REAL)1) ? horizon : (REAL)1; }
        for (unsigned int attempt = 0u; attempt < 10u; attempt++) {
            tv = umax / (REAL)(table_n - 1u);
            block[0] = (REAL)0;
            for (unsigned int k = 1u; k < table_n; k++) {
                stoch_philox(first_path + path, attempt, site_table, k, seed_lo, seed_hi, words);
                uj = stoch_uniform(words[0], words[1]);
                uv = stoch_uniform(words[2], words[3]);
TABLE
                block[k] = block[k - 1u] + table_inc[0];
            }
            if (block[table_n - 1u] >= horizon) { break; }
            if (attempt < 9u) { umax = umax * (REAL)2; }
        }
        uj = (REAL)0; uv = (REAL)0;
    }"#;

pub(super) const FRAME_SERIES_LIVE: &str = r#"        sj = (REAL)0;
        if (series_n != 0u) {
            if (series_live != 0u) {
                for (unsigned int j = 1u; j <= series_n; j++) {
                    stoch_philox(first_path + path, 0u, site_series_time, j, seed_lo, seed_hi, words);
                    REAL ratio = stoch_uniform(words[0], words[1]) * (REAL)(steps - 1u);
                    unsigned int cell = (unsigned int)ratio;
                    if ((REAL)cell < ratio) { cell += 1u; }
                    if (cell < 1u) { cell = 1u; }
                    if (cell > steps - 1u) { cell = steps - 1u; }
                    if (cell == i) {
                        stoch_philox(first_path + path, 0u, site_series, j, seed_lo, seed_hi, words);
                        gj = block[j - 1u];
                        ej = stoch_exp1(words[2], words[3]);
                        stoch_philox(first_path + path, 0u, site_series_marks, j, seed_lo, seed_hi, words);
                        uj = stoch_uniform(words[0], words[1]);
                        uv = stoch_uniform(words[2], words[3]);
SERIES
                        sj += series_size[0];
                    }
                }
                gj = (REAL)0; ej = (REAL)0; uj = (REAL)0; uv = (REAL)0;
            } else {
                sj = block[i];
            }
        }"#;

pub(super) const FRAME_TABLE_LIVE: &str = r#"        if (table_n != 0u) {
            REAL ti = dt * (REAL)i;
            while (tp < table_n && block[tp] < ti) { tp++; }
            if (tp >= table_n) {
                iv = tv * (REAL)(table_n - 1u);
            } else if (block[tp] <= block[tp - 1u]) {
                iv = tv * (REAL)tp;
            } else {
                iv = tv * (REAL)(tp - 1u) + (ti - block[tp - 1u]) / (block[tp] - block[tp - 1u]) * tv;
            }
        }"#;

pub(super) const FRAME_LIFT: &str = r#"        if (has_lift != 0u) {
LIFT
            REAL hist = (REAL)0;
            for (unsigned int l = 0u; l < lift_n; l++) { hist += lift_weight[l] * (lh[l] + lj[l]); }
            lv = lift_x0 + lift_db * lift[0] + hist + lift_fb * lift[1] * lift[2];
        }"#;

pub(super) const FRAME_HISTORY: &str = r#"        if (hist_slot != 4294967295u) {
HISTORY
            unsigned int hi = (step_first != 0u) ? i : (i - 1u);
            block[hi] = hist_in[0];
            cv = (REAL)0;
            for (unsigned int k = 0u; k <= hi; k++) { cv += curve[(INDEX)hist_slot * steps + k] * block[hi - k]; }
        }"#;
