//! The Metal Shading Language behind the fGN and sheet pipelines.
//!
//! Kept apart from the host side because it is one long string constant and
//! the two are read for different reasons: this file answers what the device
//! computes, [`super`] answers how a batch is cut and dispatched.

/// What every MSL pipeline of this crate's FFT family starts from: the
/// uniform, the 64-bit keyed draw behind it, and the radix-2 butterfly stage.
/// The sheet pipeline concatenates its own kernels behind it.
pub(crate) const MSL_COMMON: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float u01(uint x) {
    return (float(x >> 8) + 0.5f) / 16777216.0f;
}

// SplitMix64. The counter these pipelines draw on is an element of the whole
// batch, and a batch of a million 512-point paths already passes 2^30 of
// them, where a 32-bit counter hands two chunks the same stream.
inline ulong sm64(ulong x) {
    x += 0x9e3779b97f4a7c15UL;
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9UL;
    x ^= x >> 27; x *= 0x94d049bb133111ebUL;
    x ^= x >> 31;
    return x;
}

// The four uniforms of one cell: two 64-bit mixes of the cell number under
// the batch's seed, each read as two words.
inline float4 u01x4(ulong cell, uint seed) {
    ulong k = cell ^ ((ulong)seed * 0x9e3779b97f4a7c15UL);
    ulong h1 = sm64(k);
    ulong h2 = sm64(k ^ 0xd1b54a32d192ed03UL);
    return float4(u01((uint)h1), u01((uint)(h1 >> 32)),
                  u01((uint)h2), u01((uint)(h2 >> 32)));
}

kernel void fft_butterfly(
    device float* real [[buffer(0)]],
    device float* imag [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& half_stride [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint butterflies_per_batch = n / 2;
    uint batch = tid / butterflies_per_batch;
    uint local_tid = tid % butterflies_per_batch;
    uint stride = half_stride * 2;
    uint group = local_tid / half_stride;
    uint pos = local_tid % half_stride;
    uint base = batch * n;
    uint i = base + group * stride + pos;
    uint j = i + half_stride;

    float angle = -2.0f * 3.14159265358979323846f * float(pos) / float(stride);
    float tw_r = cos(angle);
    float tw_i = sin(angle);

    float tr = real[j] * tw_r - imag[j] * tw_i;
    float ti = real[j] * tw_i + imag[j] * tw_r;
    float ar = real[i];
    float ai = imag[i];

    real[i] = ar + tr;
    imag[i] = ai + ti;
    real[j] = ar - tr;
    imag[j] = ai - ti;
}
"#;

/// The fGN kernels proper: the head of the pipeline, and its two tails.
///
/// Every stage above the tile is a *pair* of butterfly stages held in
/// registers, which is one crossing of the memory system where two separate
/// dispatches were two. The pair is not bit-identical to running the stages
/// apart — the compiler contracts the multiply-add that spans them into an
/// FMA, which moves a few values by one or two units in the last place — and
/// nothing downstream depends on that, since a device's stream is its own.
pub(super) const MSL_FGN: &str = r#"
// Entry `l - 1` of row `2t + h` of the batch, where `t` is the transform and
// `h` picks its real or imaginary half. `parity` shifts the row down by one
// when the launch starts on the imaginary half of a transform: such a launch
// computes the transform in front of it as well and drops the half it does
// not own, which is what lets a batch be cut at any row and still agree with
// one launch.
inline void put(device float* out, uint out_size, uint rows, int parity,
                uint batch, uint h, uint l, float v, float scale) {
    if (l < 1u || l - 1u >= out_size) { return; }
    int row = int(2u * batch + h) - parity;
    if (row < 0 || uint(row) >= rows) { return; }
    out[uint(row) * out_size + l - 1u] = v * scale;
}

// The draw, the eigenvalue scale, the bit-reversal and the first `stages`
// butterfly stages, all inside one threadgroup's tile. Each thread draws two
// (re, im) pairs from the four uniforms of a cell of the batch, fed into two
// Box-Muller transforms, then owns one butterfly of every stage.
//
// The value belonging at bit-reversed position p is the draw of the natural
// index rev(p) — the permutation is its own inverse — so the tile can start
// from its own positions rather than from a scatter across the batch.
//
// A transform serves two rows of the batch, so its cell block sits at
// `2 * transform` blocks from the start: the row a value belongs to is then
// a function of the absolute row index alone, whatever launch produced it.
kernel void gen_tile_fft(
    device float* real [[buffer(0)]],
    device float* imag [[buffer(1)]],
    device const float* sqrt_eigs [[buffer(2)]],
    device const uint* bit_rev [[buffer(3)]],
    constant uint& traj_size [[buffer(4)]],
    constant uint& seed [[buffer(5)]],
    constant ulong& base_cell [[buffer(6)]],
    constant uint& stages [[buffer(7)]],
    threadgroup float* sh_re [[threadgroup(0)]],
    threadgroup float* sh_im [[threadgroup(1)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint half_tile [[threads_per_threadgroup]])
{
    uint tile = half_tile * 2;
    uint tile_base = gid * tile;
    uint batch = tile_base / traj_size;
    uint within = tile_base % traj_size;
    ulong cell_base = base_cell + (ulong)batch * 2 * traj_size;

    for (uint k = lid; k < tile; k += half_tile) {
        uint nat = bit_rev[within + k];
        float4 u = u01x4(cell_base + nat, seed);
        float r_a = sqrt(-2.0f * log(u.x + 1e-10f));
        float r_b = sqrt(-2.0f * log(u.z + 1e-10f));
        float eig = sqrt_eigs[nat];
        sh_re[k] = r_a * cos(6.28318530718f * u.y) * eig;
        sh_im[k] = r_b * cos(6.28318530718f * u.w) * eig;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < stages; ++s) {
        uint half_stride = 1u << s;
        uint stride = half_stride * 2;
        uint group = lid / half_stride;
        uint pos = lid % half_stride;
        uint i = group * stride + pos;
        uint j = i + half_stride;

        float angle = -2.0f * 3.14159265358979323846f * float(pos) / float(stride);
        float tw_r = cos(angle);
        float tw_i = sin(angle);

        float tr = sh_re[j] * tw_r - sh_im[j] * tw_i;
        float ti = sh_re[j] * tw_i + sh_im[j] * tw_r;
        float ar = sh_re[i];
        float ai = sh_im[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sh_re[i] = ar + tr;
        sh_im[i] = ai + ti;
        sh_re[j] = ar - tr;
        sh_im[j] = ai - ti;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint k = lid; k < tile; k += half_tile) {
        real[tile_base + k] = sh_re[k];
        imag[tile_base + k] = sh_im[k];
    }
}

// Two butterfly stages in registers. The quadruple a thread owns is closed
// under both, so the pair costs one pass over the batch instead of two.
kernel void fft_radix4(
    device float* real [[buffer(0)]],
    device float* imag [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& s0 [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint quads = n / 4;
    uint batch = tid / quads;
    uint local = tid % quads;
    uint hs0 = 1u << s0;
    uint group = local / hs0;
    uint pos = local % hs0;
    uint i0 = batch * n + group * (hs0 * 4) + pos;
    uint i1 = i0 + hs0;
    uint i2 = i1 + hs0;
    uint i3 = i2 + hs0;

    float a0r = real[i0], a0i = imag[i0];
    float a1r = real[i1], a1i = imag[i1];
    float a2r = real[i2], a2i = imag[i2];
    float a3r = real[i3], a3i = imag[i3];

    float ang0 = -2.0f * 3.14159265358979323846f * float(pos) / float(hs0 * 2);
    float w0r = cos(ang0), w0i = sin(ang0);
    float tr = a1r * w0r - a1i * w0i;
    float ti = a1r * w0i + a1i * w0r;
    float b0r = a0r + tr, b0i = a0i + ti;
    float b1r = a0r - tr, b1i = a0i - ti;
    tr = a3r * w0r - a3i * w0i;
    ti = a3r * w0i + a3i * w0r;
    float b2r = a2r + tr, b2i = a2i + ti;
    float b3r = a2r - tr, b3i = a2i - ti;

    float ang1 = -2.0f * 3.14159265358979323846f * float(pos) / float(hs0 * 4);
    float w1r = cos(ang1), w1i = sin(ang1);
    float ang2 = -2.0f * 3.14159265358979323846f * float(pos + hs0) / float(hs0 * 4);
    float w2r = cos(ang2), w2i = sin(ang2);

    tr = b2r * w1r - b2i * w1i;
    ti = b2r * w1i + b2i * w1r;
    real[i0] = b0r + tr;
    imag[i0] = b0i + ti;
    real[i2] = b0r - tr;
    imag[i2] = b0i - ti;
    tr = b3r * w2r - b3i * w2i;
    ti = b3r * w2i + b3i * w2r;
    real[i1] = b1r + tr;
    imag[i1] = b1i + ti;
    real[i3] = b1r - tr;
    imag[i3] = b1i - ti;
}

// The last two stages with the read-out folded in. Nothing of the transform
// is written back: the eight values a thread finishes are exported straight
// to the two rows the transform serves — the real half to one, the imaginary
// half to the other, both independent paths of the same law.
kernel void fft_radix4_extract(
    device const float* real [[buffer(0)]],
    device const float* imag [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& out_size [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& parity [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint hs0 = n / 4;
    uint batch = tid / hs0;
    uint pos = tid % hs0;
    uint i0 = batch * n + pos;
    uint i1 = i0 + hs0;
    uint i2 = i1 + hs0;
    uint i3 = i2 + hs0;

    float a0r = real[i0], a0i = imag[i0];
    float a1r = real[i1], a1i = imag[i1];
    float a2r = real[i2], a2i = imag[i2];
    float a3r = real[i3], a3i = imag[i3];

    float ang0 = -2.0f * 3.14159265358979323846f * float(pos) / float(hs0 * 2);
    float w0r = cos(ang0), w0i = sin(ang0);
    float tr = a1r * w0r - a1i * w0i;
    float ti = a1r * w0i + a1i * w0r;
    float b0r = a0r + tr, b0i = a0i + ti;
    float b1r = a0r - tr, b1i = a0i - ti;
    tr = a3r * w0r - a3i * w0i;
    ti = a3r * w0i + a3i * w0r;
    float b2r = a2r + tr, b2i = a2i + ti;
    float b3r = a2r - tr, b3i = a2i - ti;

    float ang1 = -2.0f * 3.14159265358979323846f * float(pos) / float(hs0 * 4);
    float w1r = cos(ang1), w1i = sin(ang1);
    float ang2 = -2.0f * 3.14159265358979323846f * float(pos + hs0) / float(hs0 * 4);
    float w2r = cos(ang2), w2i = sin(ang2);
    int par = int(parity);

    tr = b2r * w1r - b2i * w1i;
    ti = b2r * w1i + b2i * w1r;
    put(output, out_size, rows, par, batch, 0, pos, b0r + tr, scale);
    put(output, out_size, rows, par, batch, 1, pos, b0i + ti, scale);
    put(output, out_size, rows, par, batch, 0, pos + hs0 * 2, b0r - tr, scale);
    put(output, out_size, rows, par, batch, 1, pos + hs0 * 2, b0i - ti, scale);
    tr = b3r * w2r - b3i * w2i;
    ti = b3r * w2i + b3i * w2r;
    put(output, out_size, rows, par, batch, 0, pos + hs0, b1r + tr, scale);
    put(output, out_size, rows, par, batch, 1, pos + hs0, b1i + ti, scale);
    put(output, out_size, rows, par, batch, 0, pos + hs0 * 3, b1r - tr, scale);
    put(output, out_size, rows, par, batch, 1, pos + hs0 * 3, b1i - ti, scale);
}

// The same read-out for a transform of four points or fewer, where there is
// no pair of stages left above the tile to fold.
kernel void fft_butterfly_extract(
    device const float* real [[buffer(0)]],
    device const float* imag [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& out_size [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& parity [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint half_stride = n / 2;
    uint batch = tid / half_stride;
    uint pos = tid % half_stride;
    uint i = batch * n + pos;
    uint j = i + half_stride;

    float angle = -2.0f * 3.14159265358979323846f * float(pos) / float(n);
    float tw_r = cos(angle);
    float tw_i = sin(angle);
    float tr = real[j] * tw_r - imag[j] * tw_i;
    float ti = real[j] * tw_i + imag[j] * tw_r;
    float ar = real[i];
    float ai = imag[i];
    int par = int(parity);

    put(output, out_size, rows, par, batch, 0, pos, ar + tr, scale);
    put(output, out_size, rows, par, batch, 1, pos, ai + ti, scale);
    put(output, out_size, rows, par, batch, 0, pos + half_stride, ar - tr, scale);
    put(output, out_size, rows, par, batch, 1, pos + half_stride, ai - ti, scale);
}
"#;
