# Metal kernels

Rust device code for the Apple Silicon Metal backend of `stochastic-rs-stochastic`.
The host crate embeds the compiled library and ABI manifest, so downstream builds
use stable Rust and need no Metal compiler. The kernel crate is not published.

Install `cargo-metal` and its pinned compiler from the metal-oxide Git revision
in the workspace manifest. From the workspace root, regenerate artifacts with:

```sh
bash scripts/build-metal-kernels.sh
bash scripts/build-metal-kernels.sh --check
```

Commit kernel source, generated bindings, manifest, library and source checksums together.
The checksums include workspace dependencies and the lockfile; regenerate after updates.
Generation selects fast math and records the numerical mode in the artifact.
The runtime validates the library digest, ABI and device requirements when loading.
Run the stochastic crate's Metal tests on hardware after regeneration.

The fGN, sheet and Euler pipelines use these Rust kernels. Euler family formulas
are shared with the host and CUDA generator. After adding a family, regenerate
its entry names with `python3 scripts/generate-euler-kernels.py`; a family gets
one entry per jump/gamma shape that changes the draws it reads.
The Metal backend targets Apple Silicon with macOS 15 or newer.
