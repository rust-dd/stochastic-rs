use crate::euler::kernel::{cuda_language, philox_prelude};

/// Two Philox normals per complex cell, `which` 0 the real part and 1 the imaginary, scaled by the
/// embedding's eigenvalue root; `cell` is the batch-global index, so a chunk continues one launch.
const GEN_SCALE: &str = r#"
extern "C" __global__ void gen_scale_REAL(
    REAL* __restrict__ data,
    const REAL* __restrict__ sqrt_eigs,
    int traj_size, int total,
    unsigned int seed_lo, unsigned int seed_hi, unsigned long long first_cell)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    unsigned long long cell = first_cell + (unsigned long long)tid;
    unsigned int words[4];
    stoch_philox((unsigned int)cell, (unsigned int)(cell >> 32), 0u, site_fgn, seed_lo, seed_hi, words);
    REAL re = stoch_normal(words[0], words[1], words[2], words[3]);
    stoch_philox((unsigned int)cell, (unsigned int)(cell >> 32), 1u, site_fgn, seed_lo, seed_hi, words);
    REAL im = stoch_normal(words[0], words[1], words[2], words[3]);
    REAL eig = sqrt_eigs[tid % traj_size];
    data[2*tid]   = re * eig;
    data[2*tid+1] = im * eig;
}
"#;

/// The generate-and-scale kernel for `real` (`float` or `double`), after the Philox prelude.
pub(super) fn gen_scale_source(real: &'static str) -> String {
  format!(
    "{}{}",
    philox_prelude(&cuda_language(real)),
    GEN_SCALE.replace("REAL", real)
  )
}

pub(super) const EXTRACT_F32: &str = r#"
extern "C" __global__ void extract_f32(
    const float* __restrict__ data,
    float* __restrict__ output,
    int out_size, int traj_stride, float scale, int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int traj_id  = tid / out_size;
    int idx      = tid % out_size;
    output[tid]  = data[2 * (traj_id * traj_stride + idx + 1)] * scale;
}
"#;

pub(super) const EXTRACT_F64: &str = r#"
extern "C" __global__ void extract_f64(
    const double* __restrict__ data,
    double* __restrict__ output,
    int out_size, int traj_stride, double scale, int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int traj_id  = tid / out_size;
    int idx      = tid % out_size;
    output[tid]  = data[2 * (traj_id * traj_stride + idx + 1)] * scale;
}
"#;
