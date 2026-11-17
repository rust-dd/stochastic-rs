//! CUDA Poisson, gamma and compound-jump draws.

pub(super) const FRAME_COUNTS: &str = r#"        if (has_jumps != 0u) {
            REAL ell = STOCH_EXP(-jump_lambda * dt);
            REAL prod = (REAL)1;
            unsigned int cnt = 0u;
            for (unsigned int j = 0u; j < 64u; j++) {
                if ((j & 1u) == 0u) { stoch_philox(first_path + path, i, site_count, j >> 1u, seed_lo, seed_hi, w); }
                prod = prod * stoch_uniform(w[2u * (j & 1u)], w[2u * (j & 1u) + 1u]);
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
                    stoch_philox(first_path + path, i, site_gamma + gi, gamma_boost, seed_lo, seed_hi, w);
                    boost = STOCH_POW(stoch_uniform(w[0], w[1]), (REAL)1 / gsh);
                    a = a + (REAL)1;
                }
                REAL dd = a - (REAL)1 / (REAL)3;
                REAL cc = (REAL)1 / STOCH_SQRT((REAL)9 * dd);
                REAL val = dd;
                for (unsigned int j = 0u; j < 24u; j++) {
                    stoch_philox(first_path + path, i, site_gamma + gi, 2u * j, seed_lo, seed_hi, w);
                    REAL zg = stoch_normal(w[0], w[1], w[2], w[3]);
                    REAL vv = (REAL)1 + cc * zg;
                    vv = vv * vv * vv;
                    if (vv > (REAL)0) {
                        stoch_philox(first_path + path, i, site_gamma + gi, 2u * j + 1u, seed_lo, seed_hi, w);
                        REAL ug = stoch_uniform(w[0], w[1]);
                        val = dd * vv;
                        if (STOCH_FAST_LOG(ug) < (REAL)0.5 * zg * zg + dd - dd * vv + dd * STOCH_FAST_LOG(vv)) { break; }
                    }
                }
                draw = gsc * boost * val;
            }
            if (gi == 0u) { gm = draw; } else { gm2 = draw; }
        }"#;

pub(super) const FRAME_JUMP_LAWS: &str = r#"        js = (REAL)0;
        if (jump_law == 1u) {
            stoch_philox(first_path + path, i, site_sizes, 0u, seed_lo, seed_hi, w);
            REAL zj = stoch_normal(w[0], w[1], w[2], w[3]);
            js = jump_a * nj + jump_b * STOCH_SQRT(nj) * zj;
        }
        if (jump_law == 2u) {
            for (unsigned int j = 0u; j < 64u; j++) {
                if ((REAL)j >= nj) { break; }
                stoch_philox(first_path + path, i, site_sizes, j, seed_lo, seed_hi, w);
                REAL up = stoch_uniform(w[0], w[1]);
                REAL ee = stoch_exp1(w[2], w[3]);
                js += (up < jump_a) ? (ee / jump_b) : (-(ee / jump_c));
            }
        }
        if (jump_law == 3u) {
            for (unsigned int j = 0u; j < 64u; j++) {
                if ((REAL)j >= nj) { break; }
                stoch_philox(first_path + path, i, site_sizes, j, seed_lo, seed_hi, w);
                REAL xj = jump_a * STOCH_POW(stoch_uniform(w[0], w[1]), jump_b);
                if (stoch_uniform(w[2], w[3]) <= STOCH_EXP(-jump_c * xj)) { js += xj; }
            }
        }
        if (jump_law == 4u) {
            REAL jp = (REAL)1;
            for (unsigned int j = 0u; j < 64u; j++) {
                if ((REAL)j >= nj) { break; }
                stoch_philox(first_path + path, i, site_sizes, j, seed_lo, seed_hi, w);
                REAL pz = stoch_normal(w[0], w[1], w[2], w[3]);
                jp = jp * ((REAL)1 + jump_a + jump_b * pz);
            }
            js = jp - (REAL)1;
        }
        if (jump_law == 5u) {
            REAL jp = (REAL)1;
            for (unsigned int j = 0u; j < 64u; j++) {
                if ((REAL)j >= nj) { break; }
                stoch_philox(first_path + path, i, site_sizes, j, seed_lo, seed_hi, w);
                REAL up = stoch_uniform(w[0], w[1]);
                REAL ee = stoch_exp1(w[2], w[3]);
                jp = jp * ((REAL)1 + ((up < jump_a) ? (ee / jump_b) : (-(ee / jump_c))));
            }
            js = jp - (REAL)1;
        }
        if (jump_law == 6u) {
            stoch_philox(first_path + path, i, site_sizes, 0u, seed_lo, seed_hi, w);
            REAL zj = stoch_normal(w[0], w[1], w[2], w[3]);
            js = jump_a + jump_b * zj;
        }
        if (jump_law == 7u) {
            stoch_philox(first_path + path, i, site_sizes, 0u, seed_lo, seed_hi, w);
            REAL up = stoch_uniform(w[0], w[1]);
            REAL ee = stoch_exp1(w[2], w[3]);
            js = (up < jump_a) ? (ee / jump_b) : (-(ee / jump_c));
        }
        if (jump_law == 8u) {
            for (unsigned int j = 0u; j < 64u; j++) {
                if ((REAL)j >= nj) { break; }
                stoch_philox(first_path + path, i, site_sizes, j, seed_lo, seed_hi, w);
                js += (stoch_uniform(w[0], w[1]) < (REAL)0.5) ? jump_a : (-jump_a);
            }
        }
        if (jump_law == 9u) {
            stoch_philox(first_path + path, i, site_sizes, 0u, seed_lo, seed_hi, w);
            REAL va = (stoch_uniform(w[0], w[1]) - (REAL)0.5) * (REAL)3.141592653589793;
            REAL wv = stoch_exp1_accurate(w[2], w[3]);
            REAL ia = (REAL)1 / jump_a;
            REAL ratio = STOCH_COS(va - jump_a * va) / wv;
            if (ratio < (REAL)1.0e-30) { ratio = (REAL)1.0e-30; }
            // The accurate cosine and log: under a power of 1/alpha a fast intrinsic's
            // absolute error turns relative and unbounded as the angle nears +-pi/2.
            REAL xs = STOCH_SIN(jump_a * va) / STOCH_POW(STOCH_COS(va), ia) * STOCH_POW(ratio, ((REAL)1 - jump_a) * ia);
            js = jump_b * STOCH_POW(nj, ia) * xs;
        }"#;
