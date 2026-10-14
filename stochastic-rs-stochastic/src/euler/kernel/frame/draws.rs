//! CUDA Poisson, gamma and compound-jump draws.

pub(super) const FRAME_COUNTS: &str = r#"        if (has_jumps != 0u) {
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
                    REAL zg = STOCH_SQRT((REAL)-2.0 * STOCH_FAST_LOG(ga)) * STOCH_FAST_COS((REAL)6.283185307179586 * gb);
                    REAL vv = (REAL)1 + cc * zg;
                    vv = vv * vv * vv;
                    if (vv > (REAL)0) {
                        unsigned int p3 = (g ^ (668265263u + gi * 7919u + j * 104729u)) ^ (seed * 2654435761u);
                        p3 ^= p3 >> 16; p3 *= 2246822519u; p3 ^= p3 >> 13; p3 *= 3266489917u; p3 ^= p3 >> 16;
                        REAL ug = (REAL)p3 * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6;
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
            unsigned int ja = (g ^ 1103515245u) ^ (seed * 2654435761u);
            ja ^= ja >> 16; ja *= 2246822519u; ja ^= ja >> 13; ja *= 3266489917u; ja ^= ja >> 16;
            unsigned int jb = (g ^ 1013904223u) ^ (seed * 2654435761u);
            jb ^= jb >> 16; jb *= 2246822519u; jb ^= jb >> 13; jb *= 3266489917u; jb ^= jb >> 16;
            REAL ua = (REAL)ja * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6;
            REAL ub = (REAL)jb * (REAL)2.3283064e-10;
            REAL zj = STOCH_SQRT((REAL)-2.0 * STOCH_FAST_LOG(ua)) * STOCH_FAST_COS((REAL)6.283185307179586 * ub);
            js = jump_a * nj + jump_b * STOCH_SQRT(nj) * zj;
        }
        if (jump_law == 2u) {
            for (unsigned int j = 0u; j < 64u; j++) {
                if ((REAL)j >= nj) { break; }
                unsigned int ka = (g ^ (2654435761u + j * 40503u)) ^ (seed * 2654435761u);
                ka ^= ka >> 16; ka *= 2246822519u; ka ^= ka >> 13; ka *= 3266489917u; ka ^= ka >> 16;
                unsigned int kb = (g ^ (668265263u + j * 40503u)) ^ (seed * 2654435761u);
                kb ^= kb >> 16; kb *= 2246822519u; kb ^= kb >> 13; kb *= 3266489917u; kb ^= kb >> 16;
                REAL up = (REAL)ka * (REAL)2.3283064e-10;
                REAL ue = (REAL)kb * (REAL)2.3283064e-10;
                REAL ee = -STOCH_FAST_LOG((REAL)1 - ue);
                js += (up < jump_a) ? (ee / jump_b) : (-(ee / jump_c));
            }
        }
        if (jump_law == 3u) {
            for (unsigned int j = 0u; j < 64u; j++) {
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
        if (jump_law == 4u) {
            REAL jp = (REAL)1;
            for (unsigned int j = 0u; j < 64u; j++) {
                if ((REAL)j >= nj) { break; }
                unsigned int pa = (g ^ (2654435761u + j * 40503u)) ^ (seed * 2654435761u);
                pa ^= pa >> 16; pa *= 2246822519u; pa ^= pa >> 13; pa *= 3266489917u; pa ^= pa >> 16;
                unsigned int pb = (g ^ (668265263u + j * 40503u)) ^ (seed * 2654435761u);
                pb ^= pb >> 16; pb *= 2246822519u; pb ^= pb >> 13; pb *= 3266489917u; pb ^= pb >> 16;
                REAL pu1 = (REAL)pa * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6;
                REAL pu2 = (REAL)pb * (REAL)2.3283064e-10;
                REAL pz = STOCH_SQRT((REAL)-2.0 * STOCH_FAST_LOG(pu1)) * STOCH_FAST_COS((REAL)6.283185307179586 * pu2);
                jp = jp * ((REAL)1 + jump_a + jump_b * pz);
            }
            js = jp - (REAL)1;
        }
        if (jump_law == 5u) {
            REAL jp = (REAL)1;
            for (unsigned int j = 0u; j < 64u; j++) {
                if ((REAL)j >= nj) { break; }
                unsigned int ka = (g ^ (2654435761u + j * 40503u)) ^ (seed * 2654435761u);
                ka ^= ka >> 16; ka *= 2246822519u; ka ^= ka >> 13; ka *= 3266489917u; ka ^= ka >> 16;
                unsigned int kb = (g ^ (668265263u + j * 40503u)) ^ (seed * 2654435761u);
                kb ^= kb >> 16; kb *= 2246822519u; kb ^= kb >> 13; kb *= 3266489917u; kb ^= kb >> 16;
                REAL up = (REAL)ka * (REAL)2.3283064e-10;
                REAL ue = (REAL)kb * (REAL)2.3283064e-10;
                REAL ee = -STOCH_FAST_LOG((REAL)1 - ue);
                jp = jp * ((REAL)1 + ((up < jump_a) ? (ee / jump_b) : (-(ee / jump_c))));
            }
            js = jp - (REAL)1;
        }
        if (jump_law == 6u) {
            unsigned int ja = (g ^ 1103515245u) ^ (seed * 2654435761u);
            ja ^= ja >> 16; ja *= 2246822519u; ja ^= ja >> 13; ja *= 3266489917u; ja ^= ja >> 16;
            unsigned int jb = (g ^ 1013904223u) ^ (seed * 2654435761u);
            jb ^= jb >> 16; jb *= 2246822519u; jb ^= jb >> 13; jb *= 3266489917u; jb ^= jb >> 16;
            REAL ua = (REAL)ja * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6;
            REAL ub = (REAL)jb * (REAL)2.3283064e-10;
            REAL zj = STOCH_SQRT((REAL)-2.0 * STOCH_FAST_LOG(ua)) * STOCH_FAST_COS((REAL)6.283185307179586 * ub);
            js = jump_a + jump_b * zj;
        }
        if (jump_law == 7u) {
            unsigned int ka = (g ^ 2654435761u) ^ (seed * 2654435761u);
            ka ^= ka >> 16; ka *= 2246822519u; ka ^= ka >> 13; ka *= 3266489917u; ka ^= ka >> 16;
            unsigned int kb = (g ^ 668265263u) ^ (seed * 2654435761u);
            kb ^= kb >> 16; kb *= 2246822519u; kb ^= kb >> 13; kb *= 3266489917u; kb ^= kb >> 16;
            REAL up = (REAL)ka * (REAL)2.3283064e-10;
            REAL ue = (REAL)kb * (REAL)2.3283064e-10;
            REAL ee = -STOCH_FAST_LOG((REAL)1 - ue);
            js = (up < jump_a) ? (ee / jump_b) : (-(ee / jump_c));
        }
        if (jump_law == 8u) {
            for (unsigned int j = 0u; j < 64u; j++) {
                if ((REAL)j >= nj) { break; }
                unsigned int ra = (g ^ (2654435761u + j * 40503u)) ^ (seed * 2654435761u);
                ra ^= ra >> 16; ra *= 2246822519u; ra ^= ra >> 13; ra *= 3266489917u; ra ^= ra >> 16;
                js += ((REAL)ra * (REAL)2.3283064e-10 < (REAL)0.5) ? jump_a : (-jump_a);
            }
        }
        if (jump_law == 9u) {
            unsigned int sa = (g ^ 1103515245u) ^ (seed * 2654435761u);
            sa ^= sa >> 16; sa *= 2246822519u; sa ^= sa >> 13; sa *= 3266489917u; sa ^= sa >> 16;
            unsigned int sb = (g ^ 1013904223u) ^ (seed * 2654435761u);
            sb ^= sb >> 16; sb *= 2246822519u; sb ^= sb >> 13; sb *= 3266489917u; sb ^= sb >> 16;
            REAL va = ((REAL)sa * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6 - (REAL)0.5) * (REAL)3.141592653589793;
            REAL wv = -STOCH_LOG((REAL)sb * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6);
            REAL ia = (REAL)1 / jump_a;
            REAL ratio = STOCH_COS(va - jump_a * va) / wv;
            if (ratio < (REAL)1.0e-30) { ratio = (REAL)1.0e-30; }
            // The accurate cosine, not the sampling one, and the accurate
            // logarithm with it. Both feed a power of `1/alpha`, which turns
            // an approximate intrinsic's *absolute* error bound into an
            // unbounded relative one as the angle approaches ±pi/2 and then
            // amplifies it a hundredfold at `alpha = 0.01`. A draw's worth of
            // accuracy is cheap here: this runs once a jump, not once a step.
            REAL xs = STOCH_SIN(jump_a * va) / STOCH_POW(STOCH_COS(va), ia) * STOCH_POW(ratio, ((REAL)1 - jump_a) * ia);
            js = jump_b * STOCH_POW(nj, ia) * xs;
        }"#;
