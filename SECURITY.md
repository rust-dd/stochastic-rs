# Security policy

## Supported versions

Security fixes go to the latest release on crates.io and PyPI. While the 3.0 line is in release-candidate form, that is the newest `3.0.0-rc.N`; once 3.0.0 is out, the latest 3.x.

## Reporting a vulnerability

Report it privately through GitHub: [Security advisories, "Report a vulnerability"](https://github.com/rust-dd/stochastic-rs/security/advisories/new). Please do not open a public issue for it.

Include the affected crate and version, a minimal reproduction, and the impact you see. You get an acknowledgement within a week; a fix and a coordinated disclosure follow, and the advisory credits you unless you ask otherwise.

## Scope

The library crates, the Python wheels and the build and release workflows are in scope. Numerical accuracy of a model is a bug, not a vulnerability, and goes in the public issue tracker. The experimental `stochastic-rs-ai` crate and the GPU back-ends are in scope for memory-safety and supply-chain issues.
