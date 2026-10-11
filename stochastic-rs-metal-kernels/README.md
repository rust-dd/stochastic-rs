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

Commit kernel source, the manifest, library and source checksums together.
The checksums include workspace dependencies and the lockfile; regenerate after updates.
Generation uses explicit fast math to match the existing Metal pipeline.
The runtime validates the library digest, ABI and device requirements when loading.
Run the stochastic crate's Metal tests on hardware after regeneration.

The fGN pipeline is implemented here. Sheet and Euler kernels still use the
native Metal backend until their migration. The Metal backend targets Apple Silicon
with macOS 15 or newer.
