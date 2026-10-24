//! CUDA series, clock tables, lifts and history over private per-path storage.

pub(super) const FRAME_SERIES_DRAW: &str = r#"    if (series_n != 0u) {
        for (unsigned int c = 0u; c < 4u; c++) { state[c] = x0[c]; }
        for (unsigned int k = 0u; k < 512u; k++) { block[k] = (REAL)0; }
        for (unsigned int j = 1u; j <= series_n; j++) {
            unsigned int sg = ((first_path + path) * 2654435761u) ^ (j * 40503u) ^ 3266489917u;
            unsigned int q1 = (sg ^ 2654435769u) ^ (seed * 2654435761u);
            q1 ^= q1 >> 16; q1 *= 2246822519u; q1 ^= q1 >> 13; q1 *= 3266489917u; q1 ^= q1 >> 16;
            gj += -STOCH_FAST_LOG((REAL)q1 * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6);
            if (series_live != 0u) {
                block[j - 1u] = gj;
            } else {
                unsigned int q2 = (sg ^ 2246822507u) ^ (seed * 2654435761u);
                q2 ^= q2 >> 16; q2 *= 2246822519u; q2 ^= q2 >> 13; q2 *= 3266489917u; q2 ^= q2 >> 16;
                unsigned int q3 = (sg ^ 3266489909u) ^ (seed * 2654435761u);
                q3 ^= q3 >> 16; q3 *= 2246822519u; q3 ^= q3 >> 13; q3 *= 3266489917u; q3 ^= q3 >> 16;
                unsigned int q4 = (sg ^ 668265263u) ^ (seed * 2654435761u);
                q4 ^= q4 >> 16; q4 *= 2246822519u; q4 ^= q4 >> 13; q4 *= 3266489917u; q4 ^= q4 >> 16;
                unsigned int q5 = (sg ^ 374761393u) ^ (seed * 2654435761u);
                q5 ^= q5 >> 16; q5 *= 2246822519u; q5 ^= q5 >> 13; q5 *= 3266489917u; q5 ^= q5 >> 16;
                ej = -STOCH_FAST_LOG((REAL)q2 * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6);
                uj = (REAL)q3 * (REAL)2.3283064e-10;
                uv = (REAL)q4 * (REAL)2.3283064e-10;
                REAL ratio = (REAL)q5 * (REAL)2.3283064e-10 * (REAL)(steps - 1u);
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
                unsigned int tg = ((first_path + path) * 2654435761u) ^ (attempt * 668265263u) ^ (k * 40503u) ^ 2246822519u;
                unsigned int p1 = (tg ^ 2654435769u) ^ (seed * 2654435761u);
                p1 ^= p1 >> 16; p1 *= 2246822519u; p1 ^= p1 >> 13; p1 *= 3266489917u; p1 ^= p1 >> 16;
                unsigned int p2 = (tg ^ 3266489909u) ^ (seed * 2654435761u);
                p2 ^= p2 >> 16; p2 *= 2246822519u; p2 ^= p2 >> 13; p2 *= 3266489917u; p2 ^= p2 >> 16;
                uj = (REAL)p1 * (REAL)2.3283064e-10;
                uv = (REAL)p2 * (REAL)2.3283064e-10;
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
                    unsigned int sg = ((first_path + path) * 2654435761u) ^ (j * 40503u) ^ 3266489917u;
                    unsigned int q5 = (sg ^ 374761393u) ^ (seed * 2654435761u);
                    q5 ^= q5 >> 16; q5 *= 2246822519u; q5 ^= q5 >> 13; q5 *= 3266489917u; q5 ^= q5 >> 16;
                    REAL ratio = (REAL)q5 * (REAL)2.3283064e-10 * (REAL)(steps - 1u);
                    unsigned int cell = (unsigned int)ratio;
                    if ((REAL)cell < ratio) { cell += 1u; }
                    if (cell < 1u) { cell = 1u; }
                    if (cell > steps - 1u) { cell = steps - 1u; }
                    if (cell == i) {
                        unsigned int q2 = (sg ^ 2246822507u) ^ (seed * 2654435761u);
                        q2 ^= q2 >> 16; q2 *= 2246822519u; q2 ^= q2 >> 13; q2 *= 3266489917u; q2 ^= q2 >> 16;
                        unsigned int q3 = (sg ^ 3266489909u) ^ (seed * 2654435761u);
                        q3 ^= q3 >> 16; q3 *= 2246822519u; q3 ^= q3 >> 13; q3 *= 3266489917u; q3 ^= q3 >> 16;
                        unsigned int q4 = (sg ^ 668265263u) ^ (seed * 2654435761u);
                        q4 ^= q4 >> 16; q4 *= 2246822519u; q4 ^= q4 >> 13; q4 *= 3266489917u; q4 ^= q4 >> 16;
                        gj = block[j - 1u];
                        ej = -STOCH_FAST_LOG((REAL)q2 * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6);
                        uj = (REAL)q3 * (REAL)2.3283064e-10;
                        uv = (REAL)q4 * (REAL)2.3283064e-10;
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
