# Third-party notices

Original `promptboot-core` work is licensed `GPL-3.0-or-later`. The package has the combined license expression `GPL-3.0-or-later AND MIT AND (MIT OR Apache-2.0)` because it also contains the adapted libm routines described below.

The scalar `expf` and `scalbnf` code in `src/fp32_sse2.rs` is an adapted, fully in-module copy of the corresponding routines from rust-lang/libm 0.2.11. The pinned source archive is:

- URL: `https://static.crates.io/crates/libm/libm-0.2.11.crate`
- SHA-256: `8355be11b20d696c8f18f6cc018c4e372165b1fa8126cef092399c9951984ffa`
- upstream license: MIT, with contributions also offered under Apache-2.0

The complete upstream `LICENSE.txt` is preserved as `LICENSE.libm-0.2.11.txt` and as the release notice `LICENSES/libm-0.2.11.txt`. The Sun Microsystems notice attached to `expf` is preserved verbatim in the adapted source. The adaptation replaces ordinary Rust binary32 operators with explicitly rounded scalar SSE2 helpers and omits only `force_eval` floating-point exception-flag side effects, as permitted by the product's firmware-state contract. The repository-wide inventory is `THIRD_PARTY_NOTICES.md`.
