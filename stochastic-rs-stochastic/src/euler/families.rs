//! The drift / diffusion families of the Euler engine, declared once.
//!
//! A family is written in a subset that is simultaneously a valid Rust
//! expression and a valid C expression: infix arithmetic, parentheses and a
//! small vocabulary of functions. The shared [`definitions`] feed the host
//! and C generator here and the Rust Metal kernel generator in the maintainer
//! kernel crate, preserving each formula's operation order.
//!
//! `euler::family_parity` launches every declared family on the device
//! present and checks dimensions and finiteness. Its migration fixture also
//! pins exact Metal paths across every family and launch shape.
//!
//! A step or report may open with `bind name = expr;` lines before its final
//! expression. Each becomes a `let` on the host and a `const REAL` in the
//! emitted C, which is how a family names a clamped or guarded state once and
//! then reads like the host sampler it came from.
//!
//! A step or report may read `u` and `u2`, two uniforms in `[0, 1)` for the
//! step, `nj`, the number of jumps it saw, `js`, the sum of their sizes, `gm`
//! and `gm2`, one or two Gamma draws, and `ct`, the step's value of a time-varying
//! coefficient the host supplies as one value per grid point. A family that
//! never names it costs nothing for it.
//!
//! The names a step may use are fixed: `x` for the state, `dt` for the step
//! size, `dz` for the step's noise **increment** — `sqrt_dt · z` for Gaussian
//! noise, the fractional increment itself for fGN, which is what lets one
//! declaration serve both — and the family's own
//! parameters, which the generated code binds as locals from the parameter
//! buffer in declaration order. The `report` expression maps the state to what
//! the path records and is evaluated at `t = 0` as well, where no noise exists.
//!
//! A family with a `history { push (..) weights (ctK) }` clause reads `cv`,
//! the convolution of what it has pushed so far with the weights the launch
//! tabulated in curve slot `K`, indexed by lag: at the step that produces
//! grid point `i` the frame appends the pushed value `p_i` (evaluated on the
//! pre-step state, this step's noise and uniforms) and hands the step
//! `cv = Σ_{k≤i} w_k p_{i-k}`. That is the exact O(n²) scheme of a moving
//! average over the whole past — a Volterra integral on the grid, an ARMA
//! filter, a fractional stable motion — for grids up to
//! [`HISTORY_SLOTS`](crate::euler::HISTORY_SLOTS) points.
//!
//! A family with a `series { size (..) }` clause reads `sj`, the sum of the
//! terms that fell into the step's grid cell: before the steps the frame
//! draws, per path, as many terms as the launch's `series_terms()` names —
//! for each, `gj` the arrival of a unit-rate Poisson process, `ej` an
//! `Exp(1)`, `uj` and `uv` two uniforms, and a uniform arrival time on the
//! horizon — sizes each with the clause's expression and adds it to the cell
//! its time falls in. That is the shot-noise series representation of a
//! Lévy process, the sort with a rejection-free size bound, run without the
//! host's sort: cells are addressed, not ordered. Grids up to
//! [`SERIES_SLOTS`](crate::euler::SERIES_SLOTS) points. A clause written
//! `series { live size (..) }` sizes each term in the step of its cell
//! instead, where the size may read the state — a scale that follows a
//! simulated variance — and the frame keeps the arrivals per term rather
//! than the sums per cell, so the term count is what the slots then bound.
//!
//! A family with a `table { increment (..) }` clause reads `iv`, the inverse
//! of a monotone table at the step's time: before the steps the frame builds,
//! per path, a table of as many points as the launch's `table_spec()` names
//! over `[0, u_max]` — each increment from the clause's expression of the two
//! uniforms `uj`, `uv` and the spacing `tv` — trying up to ten extents, each
//! double the last, until the table reaches the horizon, exactly as the host
//! does; each
//! step then finds the first table value at or past its time and
//! interpolates the abscissa linearly. That is the inverse subordinator, the
//! first-passage clock of a positive process. Tables up to
//! [`TABLE_SLOTS`](crate::euler::TABLE_SLOTS) points; the grid is free.
//!
//! The function vocabulary is `sqrt`, `exp`, `ln`, `pow`, `abs`, `negate`,
//! `tanh`, `atan`, `sin`, `recip`, `positive`, `max`, `min`, the literal
//! `lit`, the
//! comparisons
//! `less`, `leq` and `geq`, and the branch-free `pick`. Each has a host
//! implementation in [`ops`] and a CUDA definition in `vocabulary::C_PRELUDE`,
//! the two kept together in [`vocabulary`]; the Metal kernel crate mirrors them
//! in its own `vocabulary` module. Anything outside it fails to compile on the
//! host, which is the intended way to find out that a kernel could not have run
//! it either.
//!
//! [`definitions`] stores the table in code order. The host and C generator
//! lives in [`codegen`]; `tests` pins its step and report against closed forms.

use crate::traits::FloatExt;

#[macro_use]
pub(crate) mod codegen;
pub(crate) mod vocabulary;

pub(crate) use vocabulary::ops;

#[macro_use]
mod definitions;

euler_family_definitions!(euler_families);

#[cfg(test)]
mod tests;
