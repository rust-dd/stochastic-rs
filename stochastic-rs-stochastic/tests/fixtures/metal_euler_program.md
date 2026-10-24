The expression fixture records real Volterra process paths from the handwritten
MSL Euler interpreter with fast math enabled. The metal-oxide interpreter
reproduces every recorded bit on classic Metal and Metal 4. Each row contains a
case name, point index and hexadecimal f32 bits. Every value compares bit for
bit, including NaN signs and payloads, infinities and signed zeros.

The recording contains 56 cases and 113 values. Its first line identifies the
source revision, GPU, operating system, Xcode/SDK and compilation mode.

The cases exercise all sixteen opcodes, stack depth eight, 62 instructions,
both long-program offsets, and a later step with nonzero time and state.
Both min/max operand orders cover NaNs and signed zeros. Zero cases take the
reciprocal inside the expression so the process's additions cannot hide a
wrong sign. Capture checks each case's intended result class, then writes its
raw bits without normalization. The first Volterra step has no accumulated
history.

To record a new baseline with all other GPU work stopped:

```sh
STOCHASTIC_RS_PROGRAM_FIXTURE=/absolute/fresh/path/metal_euler_program.txt \
STOCHASTIC_RS_PROGRAM_BASELINE_LABEL='revision=<sha>; metal-oxide=<sha>; GPU=<name>; OS=<version>; Xcode=<version>; SDK=<version>; math=Fast' \
cargo test -p stochastic-rs-stochastic --features metal --test euler_program_metal --locked \
  record_program_paths -- --ignored --exact --test-threads=1
```

The recorder creates its destination exclusively. Copy the captured file here
after reviewing its provenance and case coverage. Ordinary comparison never
updates it:

```sh
METAL_OXIDE_BACKEND=classic \
cargo test -p stochastic-rs-stochastic --features metal --test euler_program_metal --locked \
  program_paths_match_recording -- --ignored --exact --test-threads=1
```

Repeat with `METAL_OXIDE_BACKEND=metal4` to verify the other runtime backend.
