//! The Euler–Maruyama kernel body the native CUDA and Metal back-ends share:
//! one text, rendered per shading language, with the drift / diffusion blocks
//! spliced in from the family declarations in [`super::families`].
//!
//! The thread index `path`, the output and parameter buffers and the launch
//! arguments (`family`, `components`, `noises`, `x0`, `dt`, `sqrt_dt`,
//! `seed`, `steps`, `paths`, `first_path`, `increments`, `n_curves`,
//! `jump_lambda`, `has_jumps`, `jump_law`, `jump_a`, `jump_b`, `jump_c`,
//! `step_first`, `gamma_law`, `g1_shape`, `g1_scale`, `g1_per`,
//! `g2_shape`, `g2_scale`, `g2_per`, `has_lift`, `lift_n`, `lift_db`, `lift_fb`,
//! `lift_x0`) and the `incs`, `curve`, `lift_decay`, `lift_weight` and
//! `lift_drift_scale` buffers are bound
//! by the language-specific header around the body;
//! the body itself only uses the placeholders [`Language`] fills in. Two
//! decorrelated uniforms per noise component per step come from a
//! Murmur3-style integer hash of the cell `(first_path + path, step)` under
//! the seed, so a batch produced in chunks is bit-identical to one launch.
//! The cell number is counted in the language's 64-bit type and avalanched
//! into the 32-bit word the hashes key on: counted in 32 bits it wraps at
//! 2^31 path-steps, and two paths that far apart would then draw the same
//! noise for their whole length. Each component salts the two words with a
//! constant of its own, which is what decorrelates the components. A salt is
//! xored in rather than multiplied because a shading language may
//! constant-fold the multiplication and reject it as an overflow. The
//! uniform feeding the logarithm is clamped away from zero at `1e-6`, so a
//! device normal is truncated at `5.26σ` — `1.4e-7` of the law, four orders
//! below the `f32` the kernels compute in. When
//! `increments` is set the first component reads from `incs` instead — one
//! row of `steps - 1` increments per path — which is how a fractional
//! process reaches the same families. `increments` is a count, not a flag: a
//! path's streams sit next to each other, so path `p` reads rows
//! `p * increments` and `p * increments + 1` and one embedding feeds a
//! correlated fractional pair without the kernel binding a second buffer.
//! Interleaving per path rather than blocking per stream is what keeps a
//! chunked batch identical to one launch: a chunk covering paths
//! `first .. first + len` owns rows `increments * first ..` contiguously,
//! whatever `len` the budget chose. A row is `steps - 1` increments long, one per step the
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
//! its own candidates, so the sum is over the accepted ones. A product law
//! compounds the sizes instead, `∏ (1 + y_j) − 1`, for a step whose jump
//! multiplies the state, and a one-per-step law takes a single size whatever
//! the count, for an event-indexed path; a Rademacher law sums `±scale` over
//! the count, and a symmetric α-stable law draws the count's sum in one
//! Chambers–Mallows–Stuck step at scale `scale · n^{1/α}`. It is zero for a
//! family that declares no size law.
//!
//! A family with a `history` clause reads `cv`: the frame keeps the values the
//! family pushed so far in a per-path array of 512 slots — the grid may not
//! exceed it — and convolves them with the weight curve the launch's
//! `hist_slot` names, indexed by lag, so `cv = Σ_{k≤i} w_k p_{i−k}` at the
//! step producing point `i`. Without a clause `hist_slot` is the sentinel
//! `0xFFFFFFFF` and `cv` stays zero.
//!
//! A family with a `series` clause reads `sj`: before the steps the frame
//! draws `series_n` terms per path — a unit-rate arrival `gj`, an `Exp(1)`
//! `ej`, two uniforms `uj` and `uv`, and a uniform arrival time — sizes each
//! with the family's expression and sums it into the grid cell its time falls
//! in, an array of 512 cells the grid may not exceed; `sj` is the step's cell.
//! A `live` clause keeps each term's arrival in that array instead — the term
//! count then bounded by 512 — and the step sizes the terms of its own cell
//! against the current state, redrawing their uniforms from the same hashes.
//! Without a clause `series_n` is zero and `sj` stays zero.
//!
//! A family with a `table` clause reads `iv`: before the steps the frame
//! builds `table_n` values per path over `[0, u_max]` from the family's
//! increment expression of two uniforms and the spacing `tv`, trying up to
//! ten extents, each double the last, until the last value reaches the
//! horizon, and at
//! each step finds the first value at or past the step's time and
//! interpolates its abscissa linearly — the inverse subordinator. The table
//! holds 512 values the point count may not exceed. Without a clause
//! `table_n` is zero and `iv` stays zero.
//!
//! The history, series and table blocks share one per-path array of 512
//! values: no family declares two of them, which `families::tests` holds
//! the table to, and a thread pays for one array rather than three.
//!
//! A family may read `nj`, the number of jumps the step saw: a Poisson draw
//! with mean `jump_lambda · dt`, by Knuth's product of uniforms from a hash
//! stream of its own. It is drawn once per step, so every component of a
//! system sees the same count, and it is zero for a family that declares no
//! intensity.
//!
//! A family with a `lift` clause reads `lv`, the value its lifted component
//! takes this step under the Markov lift of a rough Volterra kernel. The
//! frame keeps the lift's two state vectors per path in fixed arrays of
//! [`super::LIFT_SLOTS`] entries, evaluates the family's drift, diffusion and
//! driving shock from the current state before the step, forms the history
//! sum `Σ w_l e_l (h_l + j_l)` over the `lift_n` nodes, and after the step
//! advances every node — exactly the recursion `VolterraLift::simulate` runs
//! on the host, node constants and boundary terms supplied by `lift_spec()`.
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
/// `REAL`, `STOCH_SQRT`, `STOCH_LOG`, `STOCH_COS`, `STOCH_FAST_LOG`,
/// `STOCH_FAST_COS`, `STOCH_SIN`, `STOCH_TANH`, `STOCH_ATAN`
/// and the
/// 64-bit buffer index type `INDEX` are the precision placeholders. Before
/// the lift and the step, the frame runs the launch's programs — the postfix
/// code of the coefficients a process wrote as expressions — on a fixed
/// stack at the step's `ct` and first state slot, and hands their values to
/// the family as `pv` and `pv2`.
///
/// Held as the blocks below rather than one literal: the body is four hundred
/// lines of C, and an edit to one of its concerns should not be an edit inside
/// a four-hundred-line string. [`render`] joins them in this order, so the
/// text a kernel is built from is exactly what a single constant gave.
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
/// What a block is there for, and so whether a kernel rendered for one shape
/// keeps it. [`Need::Always`] is the frame itself — the guard, the state, the
/// draws, the write-out; the rest is what one family or one launch reaches
/// for and another never does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Need {
  /// Every kernel carries it.
  Always,
  /// Only a family that declares a shot-noise series.
  Series,
  /// Only a family that declares a clock table.
  Table,
  /// Only a family whose step reads a coefficient program.
  Program,
  /// Only a family that declares a Markov lift.
  Lift,
  /// Only a family that declares a history window.
  History,
  /// Only a launch that draws jumps or gamma variates.
  Counts,
  /// Only a launch that draws jumps.
  Jumps,
  /// Only a family that reads one of the two spare uniforms.
  Uniforms,
}

use super::Reduce;

impl Reduce {
  /// The identity the accumulator starts at, as a C literal.
  fn identity(self) -> &'static str {
    match self {
      Reduce::Max => "-INFINITY",
      Reduce::Min => "INFINITY",
      _ => "(REAL)0",
    }
  }

  /// The fold, as a C expression over `acc[c]` and `reported[c]`.
  fn fold(self) -> &'static str {
    match self {
      Reduce::Terminal => "reported[c]",
      Reduce::Max => "acc[c] > reported[c] ? acc[c] : reported[c]",
      Reduce::Min => "acc[c] < reported[c] ? acc[c] : reported[c]",
      _ => "acc[c] + reported[c]",
    }
  }
}

/// The four splices a launch's output mode makes in the frame: the stride of
/// a path in the buffer, the accumulator's declaration, and what the entry
/// point, each step and the end of the loop do with a value.
fn reduce_splices(reduce: Reduce) -> [(&'static str, String); 5] {
  let store = |index: &str| {
    format!(
      "        for (unsigned int c = 0u; c < COMPONENTS; c++) {{ out[(INDEX)c * plane + base{index}] = acc[c]; }}"
    )
  };
  match reduce {
    Reduce::None => [
      ("STRIDE", "steps".to_string()),
      ("ACC_DECL", String::new()),
      (
        "ACC_ENTRY",
        "        for (unsigned int c = 0u; c < COMPONENTS; c++) { out[(INDEX)c * plane + base] = reported[c]; }".to_string(),
      ),
      (
        "ACC_STEP",
        "        for (unsigned int c = 0u; c < COMPONENTS; c++) { out[(INDEX)c * plane + base + i] = reported[c]; }".to_string(),
      ),
      ("ACC_STORE", String::new()),
    ],
    _ => [
      ("STRIDE", "1u".to_string()),
      (
        "ACC_DECL",
        format!(
          "    REAL acc[4];\n    for (unsigned int c = 0u; c < 4u; c++) {{ acc[c] = {}; }}",
          reduce.identity()
        ),
      ),
      (
        "ACC_ENTRY",
        format!(
          "        for (unsigned int c = 0u; c < COMPONENTS; c++) {{ acc[c] = {}; }}",
          reduce.fold()
        ),
      ),
      (
        "ACC_STEP",
        format!(
          "        for (unsigned int c = 0u; c < COMPONENTS; c++) {{ acc[c] = {}; }}",
          reduce.fold()
        ),
      ),
      ("ACC_STORE", store("")),
    ],
  }
}

pub(crate) const FRAME_BLOCKS: [(&str, Need); 14] = [
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

/// Everything a path needs before its first step: the guard, the state and
/// noise registers, the curve values at the origin, and the scratch the lift,
/// history, series and table blocks write into.
///
/// The 47-line block of [`FRAME_BLOCKS`].
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
const FRAME_LOCALS: &str = r#"    if (path >= paths) return;
    INDEX base = (INDEX)path * STRIDE;
    INDEX plane = (INDEX)paths * STRIDE;
ACC_DECL
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

/// The series preamble. A shot-noise family draws its jump times and sizes
/// for the whole path here and buckets them into `block` by the step they land
/// in, then clears the draw registers.
///
/// The 33-line block of [`FRAME_BLOCKS`].
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
const FRAME_SERIES_DRAW: &str = r#"    if (series_n != 0u) {
        for (unsigned int c = 0u; c < 4u; c++) { state[c] = x0[c]; }
        for (unsigned int k = 0u; k < 512u; k++) { block[k] = (REAL)0; }
        for (unsigned int j = 1u; j <= series_n; j++) {
            unsigned int sg = ((first_path + path) * 2654435761u) ^ (j * 40503u) ^ 3266489917u;
            unsigned int q1 = (sg ^ 2654435769u) ^ (seed * 2654435761u);
            q1 ^= q1 >> 16; q1 *= 2246822519u; q1 ^= q1 >> 13; q1 *= 3266489917u; q1 ^= q1 >> 16;
            gj += -STOCH_FAST_LOG((REAL)q1 * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6);
            if (series_live != 0u) {
                block[j] = gj;
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

/// The table preamble. A family that walks a clock of its own draws that
/// clock's increments here, doubling the horizon guess until the table spans
/// the launch's own horizon.
///
/// The 30-line block of [`FRAME_BLOCKS`].
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
const FRAME_TABLE_DRAW: &str = r#"    REAL iv = (REAL)0;
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

/// The initial state, and what the path reports for it — written unless the
/// family's first point is itself a step.
///
/// The 6-line block of [`FRAME_BLOCKS`].
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
const FRAME_ENTRY_REPORT: &str = r#"    for (unsigned int c = 0u; c < 4u; c++) { state[c] = x0[c]; reported[c] = x0[c]; }
    for (unsigned int c = 0u; c < 4u; c++) { noise[c] = (REAL)0; }
REPORT
    if (step_first == 0u) {
ACC_ENTRY
    }"#;

/// The step's own draws: the 64-bit cell key, a normal per noise component,
/// the increments a fractional pipeline supplies in their place, the curve
/// values at this step, and the two spare uniforms.
///
/// The 43-line block of [`FRAME_BLOCKS`].
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
const FRAME_NOISE: &str = r#"    for (unsigned int i = (step_first != 0u ? 0u : 1u); i < steps; i++) {
        // The path and the step enter the key as two words, never as one
        // product: `p * steps + i` in 32 bits wraps at 2^31 path-steps --
        // 2^21 paths over 1024 steps, a batch this engine can be handed --
        // and the two paths that far apart would draw the same noise for
        // their whole length. Multiplying the path by an odd constant is a
        // bijection, so no two paths share a word at any step, and the
        // avalanche below spreads the pair over the word. It is done in 32
        // bits because an Apple GPU has no 64-bit integer unit and emulates
        // every such multiply.
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

/// The two spare uniforms, `u` and `u2`: a hash each, every step, for the
/// families that draw a threshold or a wait from them. Two of the four hashes
/// a step costs, and most families read neither.
///
/// One of the [`FRAME_BLOCKS`].
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
const FRAME_UNIFORMS: &str = r#"        unsigned int hu = (g ^ 2135587861u) ^ (seed * 2654435761u);
        hu ^= hu >> 16; hu *= 2246822519u; hu ^= hu >> 13; hu *= 3266489917u; hu ^= hu >> 16;
        u = (REAL)hu * (REAL)2.3283064e-10;
        unsigned int hv = (g ^ 3266489917u) ^ (seed * 2654435761u);
        hv ^= hv >> 16; hv *= 2246822519u; hv ^= hv >> 13; hv *= 3266489917u; hv ^= hv >> 16;
        u2 = (REAL)hv * (REAL)2.3283064e-10;"#;

/// The live half of a series family: the jump whose time falls inside this
/// step, drawn now rather than bucketed, for the families that need a size at
/// the moment it arrives.
///
/// The 32-line block of [`FRAME_BLOCKS`].
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
const FRAME_SERIES_LIVE: &str = r#"        sj = (REAL)0;
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
                        gj = block[j];
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

/// The clock's value at this step, interpolated from the table the preamble
/// built.
///
/// The 11-line block of [`FRAME_BLOCKS`].
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
const FRAME_TABLE_LIVE: &str = r#"        if (table_n != 0u) {
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

/// The step's jump count, by Knuth's product of uniforms, and the gamma draws
/// a family may take beside it — Marsaglia & Tsang's squeeze, with the boost
/// that carries a shape below one.
///
/// The 52-line block of [`FRAME_BLOCKS`].
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
const FRAME_COUNTS: &str = r#"        if (has_jumps != 0u) {
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

/// The jump size, by the law the launch names. Every one keys on the same
/// cell as the step's normals, so a batch produced in chunks stays identical to
/// one launch.
///
/// The 108-line block of [`FRAME_BLOCKS`].
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
const FRAME_JUMP_LAWS: &str = r#"        js = (REAL)0;
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
            REAL wv = -STOCH_FAST_LOG((REAL)sb * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6);
            REAL ia = (REAL)1 / jump_a;
            REAL ratio = STOCH_FAST_COS(va - jump_a * va) / wv;
            if (ratio < (REAL)1.0e-30) { ratio = (REAL)1.0e-30; }
            REAL xs = STOCH_SIN(jump_a * va) / STOCH_POW(STOCH_FAST_COS(va), ia) * STOCH_POW(ratio, ((REAL)1 - jump_a) * ia);
            js = jump_b * STOCH_POW(nj, ia) * xs;
        }"#;

/// The postfix interpreter for the launch's coefficient programs: a stack
/// machine over the sixteen opcodes an `Expr` compiles to, run before the lift
/// so the step reads its coefficients as plain values.
///
/// The 31-line block of [`FRAME_BLOCKS`].
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
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
                    else if (code == 13u) { pst[psp - 1u] = STOCH_FAST_LOG(pst[psp - 1u]); }
                    else if (code == 14u) { pst[psp - 1u] = STOCH_ABS(pst[psp - 1u]); }
                    else if (code == 15u) { pst[psp - 1u] = STOCH_TANH(pst[psp - 1u]); }
                }
                if (w == 0u) { pv = pst[0]; } else { pv2 = pst[0]; }
            }
        }"#;

/// The Markov lift: the family's three coefficients, and the lifted value the
/// step reads. Left out of a kernel rendered for a family that declares no
/// lift, along with the two 176-slot histories it walks.
///
/// One of the [`FRAME_BLOCKS`].
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
const FRAME_LIFT: &str = r#"        if (has_lift != 0u) {
LIFT
            REAL hist = (REAL)0;
            for (unsigned int l = 0u; l < lift_n; l++) { hist += lift_weight[l] * (lh[l] + lj[l]); }
            lv = lift_x0 + lift_db * lift[0] + hist + lift_fb * lift[1] * lift[2];
        }"#;

/// The history window: the family's pushed value appended to this path's own
/// history and convolved with the weight curve into `cv`. Left out of a
/// kernel rendered for a family that declares no history, along with the
/// 512-slot block it walks.
///
/// One of the [`FRAME_BLOCKS`].
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
const FRAME_HISTORY: &str = r#"        if (hist_slot != 4294967295u) {
HISTORY
            unsigned int hi = (step_first != 0u) ? i : (i - 1u);
            block[hi] = hist_in[0];
            cv = (REAL)0;
            for (unsigned int k = 0u; k <= hi; k++) { cv += curve[(INDEX)hist_slot * steps + k] * block[hi - k]; }
        }"#;

/// The family's own step and report, the lift's decay, and the write-out that
/// ends an iteration.
///
/// One of the [`FRAME_BLOCKS`].
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
const FRAME_STEP: &str = r#"STEP
        if (has_lift != 0u) {
            for (unsigned int l = 0u; l < lift_n; l++) {
                lh[l] = lift_decay[l] * lh[l] + lift_drift_scale[l] * lift[0];
                lj[l] = lift_decay[l] * (lj[l] + lift[1] * lift[2]);
            }
        }
        for (unsigned int c = 0u; c < 4u; c++) { reported[c] = state[c]; }
REPORT
ACC_STEP
    }
ACC_STORE
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
  /// The logarithm and cosine the *sampling* code takes, which need not be
  /// the accurate ones.
  ///
  /// Every `STOCH_FAST_LOG` and `STOCH_FAST_COS` in the frame is a random
  /// draw — a Box-Muller normal, an exponential waiting time, a
  /// Marsaglia-Tsang rejection test — where a relative error of `2^-21`
  /// moves a normal by about one `f32` ulp and shifts a rejection boundary
  /// by as much.
  ///
  /// A family's own mathematics keeps the accurate spellings. The two are
  /// deliberately different placeholders because the family vocabulary's
  /// `#define ln(v)` expands to `STOCH_LOG`: pointing that at an approximate
  /// logarithm would take the positive-stable transform's log-domain form,
  /// which exists to survive `α = 0.01`, and hand it three fewer digits.
  ///
  /// On CUDA these are the `__logf` / `__cosf` intrinsics, which are a
  /// hardware instruction each where the accurate routines are tens; the
  /// `double` kernel has no such pair and keeps the accurate ones. Metal
  /// compiles its whole library in the relaxed mode, so its two are already
  /// this.
  pub fast_log: &'a str,
  pub fast_cos: &'a str,
  pub sin: &'a str,
  pub exp: &'a str,
  pub pow: &'a str,
  pub abs: &'a str,
  pub tanh: &'a str,
  pub atan: &'a str,
  /// The type of a buffer index; `unsigned long long` on CUDA, `uint` in MSL.
  pub index: &'a str,
  /// The 64-bit unsigned integer the per-cell RNG key is counted in;
  /// `unsigned long long` in CUDA C, `ulong` in MSL. A buffer index stays
  /// 32-bit on Metal, but the key must not: it counts path-steps, not
  /// elements of one launch's output.
  pub wide: &'a str,
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
    fast_log: "log",
    fast_cos: "cos",
    sin: "sin",
    exp: "exp",
    pow: "pow",
    abs: "abs",
    tanh: "tanh",
    atan: "atan",
    index: "uint",
    wide: "ulong",
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
      fast_log: "__logf",
      fast_cos: "__cosf",
      sin: "sinf",
      exp: "expf",
      pow: "powf",
      abs: "fabsf",
      tanh: "tanhf",
      atan: "atanf",
      index: "unsigned long long",
      wide: "unsigned long long",
    }
  } else {
    Language {
      real,
      sqrt: "sqrt",
      log: "log",
      fast_log: "log",
      fast_cos: "cos",
      cos: "cos",
      sin: "sin",
      exp: "exp",
      pow: "pow",
      abs: "fabs",
      tanh: "tanh",
      atan: "atan",
      index: "unsigned long long",
      wide: "unsigned long long",
    }
  }
}

/// The function-vocabulary defines a generated family block may use, with the
/// intrinsics of `lang` filled in. Emitted above the kernel.
pub(crate) fn prelude(lang: &Language<'_>) -> String {
  substitute(super::families::vocabulary::C_PRELUDE, lang)
}

/// Every family's block in one body, as the kernels were built until
/// September 2026 and as the rendering checks still read them.
///
/// No launch compiles this: a launch compiles [`render_for`], which carries
/// one family. What this is for is the checks that every family reaches the
/// C text at all, which is cheaper to state over one string than over 120.
/// What one launch reaches for, and so what a kernel rendered for it has to
/// carry.
///
/// The monolithic body carries every family's step behind a 120-way
/// comparison and declares the scratch every optional block needs — two
/// 176-slot lift histories and a 512-slot convolution window — whether or not
/// the launch touches them. A Tesla T4 reports the bill: 80 registers and
/// **3552 bytes of local memory per thread** in `f32`, 130 and 7104 in `f64`,
/// which is three blocks per multiprocessor and 37% occupancy, and a
/// throughput that *falls* as the batch grows because the scratch leaves L2.
/// A kernel rendered for one shape carries that family's step alone and
/// declares only the scratch it uses.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Shape {
  /// The family the kernel steps.
  pub(crate) family: u32,
  /// The family declares a Markov lift.
  pub(crate) lift: bool,
  /// The family declares a history window.
  pub(crate) history: bool,
  /// The family declares a shot-noise series.
  pub(crate) series: bool,
  /// The family declares a clock table.
  pub(crate) table: bool,
  /// The family's step reads a coefficient program.
  pub(crate) program: bool,
  /// The launch draws jumps, or names a jump law — the two are independent:
  /// a compound Poisson in count mode carries a size law with no intensity.
  pub(crate) jumps: bool,
  /// The family reads one of the two spare uniforms.
  pub(crate) uniforms: bool,
  /// Noise components the family draws, as a literal in the source: a loop
  /// bound the compiler can unroll is worth more than one it cannot.
  pub(crate) noises: u8,
  /// State components the family reports, likewise.
  pub(crate) components: u8,
  /// The launch draws gamma variates.
  pub(crate) gamma: bool,
  /// What the launch writes: every step, or one folded value a path.
  pub(crate) reduce: Reduce,
}

/// Whether `block` reads the identifier `name`.
///
/// A substring search is not enough and the difference is a wrong kernel:
/// `u` occurs inside `uj`, `uv` and `unsigned`, and it occurs *as itself*
/// inside `max(u, lit(1e-7))`, where no surrounding spaces mark it. Ten
/// device-law cases failed on that distinction before this looked at the
/// characters either side.
fn reads(block: &str, name: &str) -> bool {
  let bytes = block.as_bytes();
  let mut from = 0;
  while let Some(at) = block[from..].find(name) {
    let start = from + at;
    let end = start + name.len();
    let before = start
      .checked_sub(1)
      .map(|i| bytes[i])
      .is_none_or(|c| !c.is_ascii_alphanumeric() && c != b'_');
    let after = bytes
      .get(end)
      .is_none_or(|c| !c.is_ascii_alphanumeric() && *c != b'_');
    if before && after {
      return true;
    }
    from = end;
  }
  false
}

impl Shape {
  /// The shape of a launch: what the family declares, plus the jump and gamma
  /// laws, which belong to the process rather than to the family.
  pub(crate) fn new(family: super::families::Family, jumps: bool, gamma: bool) -> Self {
    use super::families;
    Self {
      family: family.code(),
      lift: !families::c_lift_for(family).is_empty(),
      history: !families::c_history_for(family).is_empty(),
      series: !families::c_series_for(family).is_empty(),
      table: !families::c_table_for(family).is_empty(),
      // A program value can be read anywhere the family writes C: the
      // Volterra equation takes both of them in its lift, not its step.
      program: [
        families::c_step_for(family),
        families::c_lift_for(family),
        families::c_report_for(family),
        families::c_series_for(family),
        families::c_table_for(family),
        families::c_history_for(family),
      ]
      .iter()
      .any(|block| block.contains("pv")),
      jumps,
      gamma,
      uniforms: [
        families::c_step_for(family),
        families::c_lift_for(family),
        families::c_report_for(family),
        families::c_history_for(family),
        families::c_series_for(family),
        families::c_table_for(family),
      ]
      .iter()
      .any(|block| reads(block, "u") || reads(block, "u2")),
      noises: family.noises() as u8,
      components: family.components() as u8,
      reduce: Reduce::None,
    }
  }

  /// The same shape, writing one folded value a path instead of every step.
  pub(crate) fn with_reduce(self, reduce: Reduce) -> Self {
    Self { reduce, ..self }
  }

  /// Whether the 512-slot convolution window is reachable at all.
  fn needs_block(&self) -> bool {
    self.history || self.series || self.table
  }
}

/// The kernel body for one [`Shape`]: that family's step and report spliced in
/// directly, the blocks it cannot reach left out, and the scratch it does not
/// touch declared one element wide.
pub(crate) fn render_for(lang: &Language<'_>, shape: Shape) -> String {
  use super::families;
  let family = families::Family::from_code(shape.family).expect("a declared family");
  let keep = |need: Need| -> bool {
    match need {
      Need::Always => true,
      Need::Series => shape.series,
      Need::Table => shape.table,
      Need::Program => shape.program,
      Need::Lift => shape.lift,
      Need::History => shape.history,
      Need::Counts => shape.jumps || shape.gamma,
      Need::Jumps => shape.jumps,
      Need::Uniforms => shape.uniforms,
    }
  };
  let mut body = FRAME_BLOCKS
    .iter()
    .filter(|(_, need)| keep(*need))
    .map(|(block, _)| *block)
    .collect::<Vec<_>>()
    .join("\n");
  if !shape.lift {
    body = body
      .replace("REAL lh[176];", "REAL lh[1];")
      .replace("REAL lj[176];", "REAL lj[1];");
  }
  if !shape.needs_block() {
    body = body.replace("REAL block[512];", "REAL block[1];");
  }
  // Each spliced body keeps a scope of its own. The chain got one from the
  // `if (family == …)` it no longer has, and without it the step's `const`
  // bindings and the report's collide — the same parameter, read twice.
  let scoped = |body: &str, indent: &str| -> String {
    if body.trim().is_empty() {
      return String::new();
    }
    format!("{indent}{{\n{}\n{indent}}}", body.trim_end_matches('\n'))
  };
  let body = reduce_splices(shape.reduce)
    .iter()
    .fold(body, |body, (name, code)| body.replace(name, code));
  let body = body
    .replace("NOISES", &format!("{}u", shape.noises))
    .replace("COMPONENTS", &format!("{}u", shape.components))
    .replace(
      "SERIES",
      &scoped(families::c_series_for(family), "            "),
    )
    .replace(
      "TABLE",
      &scoped(families::c_table_for(family), "                "),
    )
    .replace("LIFT", &scoped(families::c_lift_for(family), "        "))
    .replace(
      "HISTORY",
      &scoped(families::c_history_for(family), "        "),
    )
    .replace("STEP", &scoped(families::c_step_for(family), "        "))
    .replace(
      "REPORT",
      &scoped(families::c_report_for(family), "        "),
    );
  substitute(&body, lang)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn render(lang: &Language<'_>) -> String {
  let body = FRAME_BLOCKS
    .iter()
    .map(|(block, _)| *block)
    .collect::<Vec<_>>()
    .join("\n")
    .replace("NOISES", "noises")
    .replace("COMPONENTS", "components")
    .replace("SERIES", super::families::C_SERIES.trim_end_matches('\n'))
    .replace("TABLE", super::families::C_TABLE.trim_end_matches('\n'))
    .replace("LIFT", super::families::C_LIFT.trim_end_matches('\n'))
    .replace("HISTORY", super::families::C_HISTORY.trim_end_matches('\n'))
    .replace("STEP", super::families::C_STEP.trim_end_matches('\n'))
    .replace("REPORT", super::families::C_REPORT.trim_end_matches('\n'));
  substitute(&body, lang)
}

fn substitute(text: &str, lang: &Language<'_>) -> String {
  text
    .replace("INDEX", lang.index)
    .replace("U64", lang.wide)
    .replace("STOCH_SQRT", lang.sqrt)
    // The frame's own draws first: `STOCH_FAST_LOG` is a Box-Muller normal
    // or an exponential waiting time, where `STOCH_LOG` is whatever a family
    // wrote as `ln`, and those are two different accuracy requirements.
    .replace("STOCH_FAST_LOG", lang.fast_log)
    .replace("STOCH_FAST_COS", lang.fast_cos)
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

  /// Every family renders to a kernel of its own, in every language, with no
  /// placeholder left and no other family's step inside it.
  ///
  /// The specialised body is what a launch compiles; a family that renders
  /// wrong here fails inside a driver's compiler on a machine that has the
  /// device, which is exactly the failure a Mac cannot see for CUDA and a
  /// CUDA box cannot see for Metal.
  #[test]
  fn every_family_renders_a_kernel_of_its_own() {
    for (name, lang) in languages() {
      for family in super::super::families::Family::ALL {
        let shape = super::Shape::new(*family, true, true);
        let source = super::render_for(&lang, shape);
        assert!(
          !source.contains("if (family =="),
          "{name}/{family:?}: a specialised kernel still compares the family"
        );
        for placeholder in [
          "STOCH_", "INDEX", "U64", "STEP", "REPORT", "LIFT", "HISTORY",
        ] {
          assert!(
            !source.contains(placeholder),
            "{name}/{family:?}: the {placeholder} placeholder survived"
          );
        }
        let balance = source.chars().fold(0i32, |d, c| match c {
          '{' => d + 1,
          '}' => d - 1,
          _ => d,
        });
        assert_eq!(balance, 0, "{name}/{family:?}: unbalanced body");
      }
    }
  }

  /// A family that declares no lift, history, series or table carries none of
  /// their scratch: the two 176-slot lift histories and the 512-slot window
  /// are what a T4 reports as 3552 bytes of local memory per thread.
  #[test]
  fn a_plain_family_declares_no_scratch_it_cannot_reach() {
    let lang = metal_language();
    let plain = super::Shape::new(
      super::super::families::Family::GeometricBrownian,
      false,
      false,
    );
    let source = super::render_for(&lang, plain);
    for scratch in ["lh[176]", "lj[176]", "block[512]"] {
      assert!(
        !source.contains(scratch),
        "a plain family still declares {scratch}"
      );
    }
    assert!(
      !source.contains("jump_law == 9u") && !source.contains("gi < gamma_law"),
      "a launch with no jumps still carries the jump laws"
    );
    let full = super::render_for(
      &lang,
      super::Shape::new(
        super::super::families::Family::GeometricBrownian,
        true,
        true,
      ),
    );
    assert!(
      full.len() > source.len(),
      "the jump laws must cost something, or dropping them buys nothing"
    );
  }

  /// The blocks join into one body, not eleven fragments.
  ///
  /// Each block opens braces a later one closes — the step loop's head is in
  /// [`FRAME_NOISE`] and its tail in [`FRAME_STEP`] — so a block dropped from
  /// [`FRAME_BLOCKS`] or moved out of order leaves the body unbalanced. That
  /// is a failure no machine without a device would otherwise see, since the
  /// text only reaches a compiler when a kernel is built.
  #[test]
  fn the_frame_blocks_join_into_a_balanced_body() {
    let joined = super::FRAME_BLOCKS
      .iter()
      .map(|(block, _)| *block)
      .collect::<Vec<_>>()
      .join("\n");
    let mut depth = 0i32;
    let mut lowest = 0i32;
    for c in joined.chars() {
      match c {
        '{' => depth += 1,
        '}' => depth -= 1,
        _ => {}
      }
      lowest = lowest.min(depth);
    }
    assert_eq!(
      depth, 0,
      "the frame opens and closes its own braces — the kernel's function brace \
       belongs to the language header and its close is appended by the \
       back-end — and this body is left at {depth}"
    );
    assert_eq!(
      lowest, 0,
      "a block closes a brace no earlier block opened; the order is wrong"
    );
    // Every block closes what it opens, so a kernel rendered for one family
    // can leave any of them out — except the two that frame the step loop,
    // which is why they are the two a specialised kernel always keeps.
    for (i, (block, _)) in super::FRAME_BLOCKS.iter().enumerate() {
      let balance = block.chars().fold(0i32, |d, c| match c {
        '{' => d + 1,
        '}' => d - 1,
        _ => d,
      });
      // The two that frame the step loop, by their place in the list.
      let want = match i {
        4 => 1,
        13 => -1,
        _ => 0,
      };
      assert_eq!(
        balance, want,
        "block {i} leaves {balance} braces open, not {want}; an optional block \
         that does not close what it opens cannot be left out"
      );
    }
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
      assert!(
        !source.contains("U64"),
        "{name}: the wide-integer placeholder survived rendering"
      );
    }
  }

  /// The path and the step reach the key as two words, never as one product.
  ///
  /// `p * steps + i` in 32 bits wraps at 2^31 path-steps — 2^21 paths over
  /// 1024 steps — and the two paths that far apart draw the same noise from
  /// there on, which no suite can catch: the batch that shows it is 8 GB of
  /// output. Two words never collide, because multiplying the path by an odd
  /// constant is a bijection. Pinned here because the alternative is a
  /// correctness bug no test can reach.
  #[test]
  fn the_cell_key_takes_the_path_and_the_step_apart() {
    for (name, lang) in languages() {
      let source = render(&lang);
      assert!(
        source.contains("unsigned int g = (first_path + path) * 2654435761u;"),
        "{name}: the path does not enter the key through an odd multiplier"
      );
      assert!(
        !source.contains("(first_path + path) * steps"),
        "{name}: the key is a product of the path and the step count"
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
