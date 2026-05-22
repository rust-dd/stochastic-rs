/// Fused generate + scale NVRTC kernel (Philox-2x32-10 RNG + Box-Muller, f32).
/// Generates two standard normals per thread, scales by sqrt_eigenvalue,
/// and writes interleaved complex output.
pub(super) const GEN_SCALE_F32: &str = r#"
extern "C" __global__ void gen_scale_f32(
    float* __restrict__ data,
    const float* __restrict__ sqrt_eigs,
    int traj_size, int total,
    unsigned long long seed, unsigned long long seq)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    /* Philox-2x32-10 */
    unsigned int lo = (unsigned int)((unsigned long long)tid + seq);
    unsigned int hi = (unsigned int)(((unsigned long long)tid + seq) >> 32);
    unsigned int k  = (unsigned int)seed;
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        unsigned long long p = (unsigned long long)0xD2511F53u * lo;
        lo = ((unsigned int)(p >> 32)) ^ hi ^ k;
        hi = (unsigned int)p;
        k += 0x9E3779B9u;
    }

    float u1 = (lo + 0.5f) * 2.3283064365386963e-10f;
    float u2 = (hi + 0.5f) * 2.3283064365386963e-10f;
    float r  = sqrtf(-2.0f * logf(u1));
    float sn, cs;
    __sincosf(6.283185307179586f * u2, &sn, &cs);

    float eig = sqrt_eigs[tid % traj_size];
    data[2*tid]   = r * cs * eig;
    data[2*tid+1] = r * sn * eig;
}
"#;

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

pub(super) const GEN_SCALE_F64: &str = r#"
extern "C" __global__ void gen_scale_f64(
    double* __restrict__ data,
    const double* __restrict__ sqrt_eigs,
    int traj_size, int total,
    unsigned long long seed, unsigned long long seq)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    unsigned int lo = (unsigned int)((unsigned long long)tid + seq);
    unsigned int hi = (unsigned int)(((unsigned long long)tid + seq) >> 32);
    unsigned int k  = (unsigned int)seed;
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        unsigned long long p = (unsigned long long)0xD2511F53u * lo;
        lo = ((unsigned int)(p >> 32)) ^ hi ^ k;
        hi = (unsigned int)p;
        k += 0x9E3779B9u;
    }

    double u1 = ((double)lo + 0.5) * 2.3283064365386963e-10;
    double u2 = ((double)hi + 0.5) * 2.3283064365386963e-10;
    double r  = sqrt(-2.0 * log(u1));
    double angle = 6.283185307179586 * u2;

    double eig = sqrt_eigs[tid % traj_size];
    data[2*tid]   = r * cos(angle) * eig;
    data[2*tid+1] = r * sin(angle) * eig;
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
