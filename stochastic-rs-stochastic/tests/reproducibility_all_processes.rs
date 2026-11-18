//! Every concrete `ProcessExt` implementor under a `Deterministic` seed: fresh instances agree to
//! the bit, another seed differs, `sample_par` ignores the pool size. A new type adds a `guard!`.

#[path = "reproducibility_all_processes/common.rs"]
mod common;

// 9 of the 129 guards.
#[path = "reproducibility_all_processes/autoregressive.rs"]
mod autoregressive;
// 4 guards.
#[path = "reproducibility_all_processes/correlation.rs"]
mod correlation;
// 36 guards.
#[path = "reproducibility_all_processes/diffusion.rs"]
mod diffusion;
// 15 guards.
#[path = "reproducibility_all_processes/interest.rs"]
mod interest;
// 16 guards.
#[path = "reproducibility_all_processes/jump.rs"]
mod jump;
// 6 guards.
#[path = "reproducibility_all_processes/noise.rs"]
mod noise;
// 19 guards.
#[path = "reproducibility_all_processes/process.rs"]
mod process;
// 4 guards.
#[path = "reproducibility_all_processes/rough.rs"]
mod rough;
// 1 guard.
#[path = "reproducibility_all_processes/sheet.rs"]
mod sheet;
// 15 guards.
#[path = "reproducibility_all_processes/volatility.rs"]
mod volatility;
// 4 guards.
#[path = "reproducibility_all_processes/volterra.rs"]
mod volterra;
