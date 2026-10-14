The `euler::family_parity::migration` tests reuse all existing family probes.
They cover every family under all four jump/gamma shape combinations, three raw
u64 seeds, all 64 grid points in every returned plane, three paths starting at
global path 3, both first-point conventions, and near-u32 path offsets for GBM
and the four-noise correlated GBM family. Every shape is checked for exact
repeatability and split-launch identity before recording. The live-series
family also pins 1 and 511 terms; a separate test checks 512 terms against an
independent reference.

The fixture uses `STOEULR1`, a little-endian family count, then ordered cases:
a u32 label length, UTF-8 label, u32 value count and f32 bits in plane/path/step
order. It records every output bit, including signs of zero and any non-finite
results from diagnostic shape combinations. Comparison accepts no tolerance.

The fixture contains 1,579 cases and 2,152,832 bytes from the Euler kernels in
`MathMode::Fast` (relaxed math, fast fp32 functions, fast contraction). Its
`.meta` sibling records revision, GPU, operating system, Xcode/SDK and math
mode. Classic Metal and Metal 4 produce identical bytes. Fast contraction lets
the GPU compiler group multiplies and adds from each kernel's structure, so a
kernel change can move final bits; record a new fixture only after the family
law and edge tests pass.

To record a new fixture with all other GPU work stopped:

```sh
STOCHASTIC_RS_EULER_BASELINE_LABEL='revision=<sha>; metal-oxide=<sha>; GPU=<name>; OS=<version>; Xcode=<version>; SDK=<version>; math=Fast' \
cargo test -p stochastic-rs-stochastic --features metal --lib --locked \
  euler::family_parity::migration::record_euler_paths -- --ignored --exact --test-threads=1
```

The default destinations are `metal_euler.bin` and `metal_euler.meta` in this
directory. Both are created exclusively. `STOCHASTIC_RS_EULER_FIXTURE` selects
a different binary path and the metadata uses its `.meta` sibling.
An ordinary comparison never updates either file.

```sh
METAL_OXIDE_BACKEND=classic \
cargo test -p stochastic-rs-stochastic --features metal --lib --locked \
  euler::family_parity::migration::euler_paths_match_recording -- --ignored --exact --test-threads=1
```

Repeat with `METAL_OXIDE_BACKEND=metal4`. These cases exercise the Euler frame
with hashed noise. Fractional increment handoff, actual process seed
advancement, all nine jump laws, and the 512-term live-series correction need
their dedicated tests.
