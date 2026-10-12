# Metal kernels

Rust device code for the Apple Silicon Metal backend of `stochastic-rs-stochastic`.
The host crate embeds the compiled library and ABI manifest, so downstream builds
use stable Rust and need no Metal compiler. The kernel crate is not published.

With the `metal` feature, the stochastic crate's build script keeps the embedded
artifact current. `cargo build -p stochastic-rs-stochastic --features metal`
rewrites the Euler entry names from the family definitions and rebuilds
`stochastic-rs-stochastic/src/device/metal/artifact` when a recorded input
changed: a kernel source, a family definition or the metal-oxide revision in
`Cargo.lock`. Rebuilding needs the compiler nightly named in metal-oxide (with
`rustc-dev`, `rust-src` and `llvm-tools`) and Xcode's Metal toolchain; the
compiler is built from the pinned metal-oxide checkout on first use. An unchanged
artifact needs neither. CI sets `METAL_OXIDE_ARTIFACTS=locked`, which fails
instead of rebuilding.

Commit kernel sources, Euler entries and the artifact (bindings, manifest,
library and `inputs.json`) together.
Generation selects fast math and records the numerical mode in the artifact.
The runtime validates the library digest, ABI and device requirements when loading.
Run the stochastic crate's Metal tests on hardware after regeneration.

The fGN, sheet and Euler pipelines use these Rust kernels. Euler family formulas
are shared with the host and CUDA generator. The build script derives the entry
names; a family gets one entry per jump/gamma shape that changes the draws it reads.
The Metal backend targets Apple Silicon with macOS 15 or newer.
