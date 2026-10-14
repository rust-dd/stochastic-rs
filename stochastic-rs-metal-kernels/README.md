# Metal kernels

Rust device code for the Apple Silicon Metal backend of `stochastic-rs-stochastic`,
which launches it through
`#[metal_oxide::kernels(path = "../stochastic-rs-metal-kernels", math = "fast")]`.
The host crate embeds the compiled library, so downstream builds use stable Rust
and need no Metal compiler. The kernel crate is not published.

With the `metal` feature, the stochastic crate's build script rewrites the Euler
entry names from the family definitions, and the kernel macro builds the kernels
and saves their artifact in `stochastic-rs-stochastic/metal/` when a recorded input
changed: a kernel source, a family definition or the metal-oxide revision in
`Cargo.lock`. Building needs the compiler nightly named in metal-oxide (with
`rustc-dev`, `rust-src` and `llvm-tools`) and Xcode's Metal toolchain; the
compiler is built from the pinned metal-oxide checkout on first use. Without
them, and for the crate's dependents, the saved artifact stands in while it is
current. CI sets `METAL_OXIDE_ARTIFACTS=locked`, which fails on stale entries or
a stale artifact instead of rebuilding.

Commit kernel sources, Euler entries and `stochastic-rs-stochastic/metal/` together.
The kernels build with fast math, and the artifact records the numerical mode.
The runtime validates the library digest, ABI and device requirements when loading.
Run the stochastic crate's Metal tests on hardware after regeneration.

The fGN, sheet and Euler pipelines use these Rust kernels. Euler family formulas
are shared with the host and CUDA generator. The build script derives the entry
names; a family gets one entry per jump/gamma shape that changes the draws it reads.
The Metal backend targets Apple Silicon with macOS 15 or newer.
