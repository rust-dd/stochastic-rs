//! Every concrete `ProcessExt` implementor under a `Deterministic` seed: fresh instances agree to
//! the bit, another seed differs, `sample_par` ignores the pool size. A new type adds a `guard!`.

#[path = "reproducibility_all_processes/common.rs"]
mod common;

#[path = "reproducibility_all_processes/autoregressive.rs"]
mod autoregressive;
#[path = "reproducibility_all_processes/correlation.rs"]
mod correlation;
#[path = "reproducibility_all_processes/diffusion.rs"]
mod diffusion;
#[path = "reproducibility_all_processes/interest.rs"]
mod interest;
#[path = "reproducibility_all_processes/jump.rs"]
mod jump;
#[path = "reproducibility_all_processes/noise.rs"]
mod noise;
#[path = "reproducibility_all_processes/process.rs"]
mod process;
#[path = "reproducibility_all_processes/rough.rs"]
mod rough;
#[path = "reproducibility_all_processes/sheet.rs"]
mod sheet;
#[path = "reproducibility_all_processes/volatility.rs"]
mod volatility;
#[path = "reproducibility_all_processes/volterra.rs"]
mod volterra;
