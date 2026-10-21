# Third-party Rust dependency licenses

> **Generated file — do not edit by hand.** Regenerate with:
> ```bash
> cargo metadata --format-version 1 --all-features \
>   | python3 tools/gen_third_party_licenses.py > THIRD_PARTY_LICENSES.md
> ```
>
> Only **crates.io / git dependencies** are listed. Vendored CUDA sources are
> attributed in `NOTICE` and `VENDOR.list.md`.
> Audit result: **no GPL/AGPL dependency**; `r-efi` merely offers
> `LGPL-2.1-or-later` as one of three choices and is absent from the
> x86_64-linux dependency tree.

**146** packages (workspace crates excluded).

| crate | version | license | upstream |
|---|---|---|---|
| `adler2` | 2.0.1 | 0BSD OR MIT OR Apache-2.0 | gh:oyvindln/adler2 |
| `ahash` | 0.8.12 | MIT OR Apache-2.0 | gh:tkaitchuck/ahash |
| `aho-corasick` | 1.1.5 | Unlicense OR MIT | gh:BurntSushi/aho-corasick |
| `allocator-api2` | 0.2.21 | MIT OR Apache-2.0 | gh:zakarumych/allocator-api2 |
| `autocfg` | 1.5.1 | Apache-2.0 OR MIT | gh:cuviper/autocfg |
| `base64` | 0.13.1 | MIT/Apache-2.0 | gh:marshallpierce/rust-base64 |
| `base64` | 0.22.1 | MIT OR Apache-2.0 | gh:marshallpierce/rust-base64 |
| `bitflags` | 2.13.2 | MIT OR Apache-2.0 | gh:bitflags/bitflags |
| `bytemuck` | 1.25.2 | Zlib OR Apache-2.0 OR MIT | gh:Lokathor/bytemuck |
| `byteorder-lite` | 0.1.0 | Unlicense OR MIT | gh:image-rs/byteorder-lite |
| `bytes` | 1.12.1 | MIT | gh:tokio-rs/bytes |
| `castaway` | 0.2.4 | MIT | gh:sagebind/castaway |
| `cc` | 1.4.7 | MIT OR Apache-2.0 | gh:rust-lang/cc-rs |
| `cfg-if` | 1.0.5 | MIT OR Apache-2.0 | gh:rust-lang/cfg-if |
| `compact_str` | 0.9.1 | MIT | gh:ParkMyCar/compact_str |
| `crc32fast` | 1.5.3 | MIT OR Apache-2.0 | gh:srijs/rust-crc32fast |
| `crossbeam-deque` | 0.8.8 | MIT OR Apache-2.0 | gh:crossbeam-rs/crossbeam |
| `crossbeam-epoch` | 0.9.21 | MIT OR Apache-2.0 | gh:crossbeam-rs/crossbeam |
| `crossbeam-utils` | 0.8.23 | MIT OR Apache-2.0 | gh:crossbeam-rs/crossbeam |
| `crunchy` | 0.2.4 | MIT | gh:eira-fransham/crunchy |
| `cudarc` | 0.19.9 | MIT OR Apache-2.0 | gh:chelsea0x3b/cudarc |
| `darling` | 0.20.11 | MIT | gh:TedDriggs/darling |
| `darling_core` | 0.20.11 | MIT | gh:TedDriggs/darling |
| `darling_macro` | 0.20.11 | MIT | gh:TedDriggs/darling |
| `dary_heap` | 0.3.9 | MIT OR Apache-2.0 | gh:hanmertens/dary_heap |
| `derive_builder` | 0.20.2 | MIT OR Apache-2.0 | gh:colin-kiegel/rust-derive-builder |
| `derive_builder_core` | 0.20.2 | MIT OR Apache-2.0 | gh:colin-kiegel/rust-derive-builder |
| `derive_builder_macro` | 0.20.2 | MIT OR Apache-2.0 | gh:colin-kiegel/rust-derive-builder |
| `either` | 1.18.0 | MIT OR Apache-2.0 | gh:rayon-rs/either |
| `equivalent` | 1.0.2 | Apache-2.0 OR MIT | gh:indexmap-rs/equivalent |
| `errno` | 0.3.14 | MIT OR Apache-2.0 | gh:lambda-fairy/rust-errno |
| `esaxx-rs` | 0.1.10 | Apache-2.0 | gh:Narsil/esaxx-rs |
| `fastrand` | 2.5.0 | Apache-2.0 OR MIT | gh:smol-rs/fastrand |
| `fdeflate` | 0.3.7 | MIT OR Apache-2.0 | gh:image-rs/fdeflate |
| `find-msvc-tools` | 0.1.13 | MIT OR Apache-2.0 | gh:rust-lang/cc-rs |
| `flate2` | 1.1.10 | MIT OR Apache-2.0 | gh:rust-lang/flate2-rs |
| `fnv` | 1.0.7 | Apache-2.0 / MIT | gh:servo/rust-fnv |
| `foldhash` | 0.2.0 | Zlib | gh:orlp/foldhash |
| `futures-core` | 0.3.34 | MIT OR Apache-2.0 | gh:rust-lang/futures-rs |
| `futures-macro` | 0.3.34 | MIT OR Apache-2.0 | gh:rust-lang/futures-rs |
| `futures-task` | 0.3.34 | MIT OR Apache-2.0 | gh:rust-lang/futures-rs |
| `futures-util` | 0.3.34 | MIT OR Apache-2.0 | gh:rust-lang/futures-rs |
| `getrandom` | 0.3.4 | MIT OR Apache-2.0 | gh:rust-random/getrandom |
| `half` | 2.7.1 | MIT OR Apache-2.0 | gh:VoidStarKat/half-rs |
| `hashbrown` | 0.16.1 | MIT OR Apache-2.0 | gh:rust-lang/hashbrown |
| `hashbrown` | 0.17.1 | MIT OR Apache-2.0 | gh:rust-lang/hashbrown |
| `ident_case` | 1.0.1 | MIT/Apache-2.0 | gh:TedDriggs/ident_case |
| `image` | 0.25.10 | MIT OR Apache-2.0 | gh:image-rs/image |
| `indexmap` | 2.14.2 | Apache-2.0 OR MIT | gh:indexmap-rs/indexmap |
| `itertools` | 0.14.0 | MIT OR Apache-2.0 | gh:rust-itertools/itertools |
| `itoa` | 1.0.18 | MIT OR Apache-2.0 | gh:dtolnay/itoa |
| `libc` | 0.2.189 | MIT OR Apache-2.0 | gh:rust-lang/libc |
| `libloading` | 0.9.0 | ISC | gh:nagisa/rust_libloading |
| `linux-raw-sys` | 0.12.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | gh:sunfishcode/linux-raw-sys |
| `lock_api` | 0.4.14 | MIT OR Apache-2.0 | gh:Amanieu/parking_lot |
| `log` | 0.4.34 | MIT OR Apache-2.0 | gh:rust-lang/log |
| `macro_rules_attribute` | 0.2.3 | Apache-2.0 OR MIT OR Zlib | gh:danielhenrymantilla/macro_rules_attribute-rs |
| `macro_rules_attribute-proc_macro` | 0.2.3 | Apache-2.0 OR MIT OR Zlib | gh:danielhenrymantilla/macro_rules_attribute-rs |
| `memchr` | 2.8.3 | Unlicense OR MIT | gh:BurntSushi/memchr |
| `memo-map` | 0.3.4 | Apache-2.0 | gh:mitsuhiko/memo-map |
| `minijinja` | 2.24.0 | Apache-2.0 | gh:mitsuhiko/minijinja |
| `minijinja-contrib` | 2.24.0 | Apache-2.0 | gh:mitsuhiko/minijinja |
| `minimal-lexical` | 0.2.1 | MIT/Apache-2.0 | gh:Alexhuszagh/minimal-lexical |
| `miniz_oxide` | 0.8.9 | MIT OR Zlib OR Apache-2.0 | gh:Frommi/miniz_oxide/tree/master/miniz_oxide |
| `miniz_oxide` | 0.9.1 | MIT OR Zlib OR Apache-2.0 | gh:Frommi/miniz_oxide/tree/master/miniz_oxide |
| `mio` | 1.2.3 | MIT | gh:tokio-rs/mio |
| `monostate` | 0.1.18 | MIT OR Apache-2.0 | gh:dtolnay/monostate |
| `monostate-impl` | 0.1.18 | MIT OR Apache-2.0 | gh:dtolnay/monostate |
| `moxcms` | 0.8.1 | BSD-3-Clause OR Apache-2.0 | gh:awxkee/moxcms.git |
| `nom` | 7.1.3 | MIT | gh:Geal/nom |
| `num-traits` | 0.2.19 | MIT OR Apache-2.0 | gh:rust-num/num-traits |
| `once_cell` | 1.21.4 | MIT OR Apache-2.0 | gh:matklad/once_cell |
| `onig` | 6.5.3 | MIT | gh:iwillspeak/rust-onig |
| `onig_sys` | 69.9.3 | MIT | gh:rust-onig/rust-onig |
| `parking_lot` | 0.12.5 | MIT OR Apache-2.0 | gh:Amanieu/parking_lot |
| `parking_lot_core` | 0.9.12 | MIT OR Apache-2.0 | gh:Amanieu/parking_lot |
| `paste` | 1.0.15 | MIT OR Apache-2.0 | gh:dtolnay/paste |
| `pastey` | 0.2.3 | MIT OR Apache-2.0 | gh:as1100k/pastey |
| `pin-project-lite` | 0.2.17 | Apache-2.0 OR MIT | gh:taiki-e/pin-project-lite |
| `pkg-config` | 0.3.34 | MIT OR Apache-2.0 | gh:rust-lang/pkg-config-rs |
| `png` | 0.18.1 | MIT OR Apache-2.0 | gh:image-rs/image-png |
| `ppv-lite86` | 0.2.21 | MIT OR Apache-2.0 | gh:cryptocorrosion/cryptocorrosion |
| `proc-macro2` | 1.0.107 | MIT OR Apache-2.0 | gh:dtolnay/proc-macro2 |
| `pxfm` | 0.1.30 | BSD-3-Clause OR Apache-2.0 | gh:awxkee/pxfm |
| `quote` | 1.0.47 | MIT OR Apache-2.0 | gh:dtolnay/quote |
| `r-efi` | 5.3.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later | gh:r-efi/r-efi |
| `rand` | 0.9.5 | MIT OR Apache-2.0 | gh:rust-random/rand |
| `rand_chacha` | 0.9.0 | MIT OR Apache-2.0 | gh:rust-random/rand |
| `rand_core` | 0.9.5 | MIT OR Apache-2.0 | gh:rust-random/rand |
| `rayon` | 1.12.0 | MIT OR Apache-2.0 | gh:rayon-rs/rayon |
| `rayon-cond` | 0.4.0 | Apache-2.0/MIT | gh:cuviper/rayon-cond |
| `rayon-core` | 1.13.0 | MIT OR Apache-2.0 | gh:rayon-rs/rayon |
| `redox_syscall` | 0.5.18 | MIT | https://gitlab.redox-os.org/redox-os/syscall |
| `regex` | 1.13.1 | MIT OR Apache-2.0 | gh:rust-lang/regex |
| `regex-automata` | 0.4.18 | MIT OR Apache-2.0 | gh:rust-lang/regex |
| `regex-syntax` | 0.8.11 | MIT OR Apache-2.0 | gh:rust-lang/regex |
| `rustix` | 1.1.5 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | gh:bytecodealliance/rustix |
| `rustversion` | 1.0.23 | MIT OR Apache-2.0 | gh:dtolnay/rustversion |
| `ryu` | 1.0.23 | Apache-2.0 OR BSL-1.0 | gh:dtolnay/ryu |
| `safetensors` | 0.8.0 | Apache-2.0 | gh:huggingface/safetensors |
| `scopeguard` | 1.2.0 | MIT OR Apache-2.0 | gh:bluss/scopeguard |
| `serde` | 1.0.229 | MIT OR Apache-2.0 | gh:serde-rs/serde |
| `serde_core` | 1.0.229 | MIT OR Apache-2.0 | gh:serde-rs/serde |
| `serde_derive` | 1.0.229 | MIT OR Apache-2.0 | gh:serde-rs/serde |
| `serde_json` | 1.0.151 | MIT OR Apache-2.0 | gh:serde-rs/json |
| `serde_spanned` | 0.6.9 | MIT OR Apache-2.0 | gh:toml-rs/toml |
| `shlex` | 2.0.1 | MIT OR Apache-2.0 | gh:comex/rust-shlex |
| `signal-hook-registry` | 1.4.8 | MIT OR Apache-2.0 | gh:vorner/signal-hook |
| `simd-adler32` | 0.3.10 | MIT | gh:mcountryman/simd-adler32 |
| `slab` | 0.4.12 | MIT | gh:tokio-rs/slab |
| `smallvec` | 1.16.1 | MIT OR Apache-2.0 | gh:servo/rust-smallvec |
| `socket2` | 0.6.5 | MIT OR Apache-2.0 | gh:rust-lang/socket2 |
| `spm_precompiled` | 0.1.4 | Apache-2.0 | gh:huggingface/spm_precompiled |
| `static_assertions` | 1.1.0 | MIT OR Apache-2.0 | gh:nvzqz/static-assertions-rs |
| `strsim` | 0.11.1 | MIT | gh:rapidfuzz/strsim-rs |
| `syn` | 2.0.119 | MIT OR Apache-2.0 | gh:dtolnay/syn |
| `syn` | 3.0.6 | MIT OR Apache-2.0 | gh:dtolnay/syn |
| `tempfile` | 3.27.0 | MIT OR Apache-2.0 | gh:Stebalien/tempfile |
| `thiserror` | 1.0.69 | MIT OR Apache-2.0 | gh:dtolnay/thiserror |
| `thiserror` | 2.0.20 | MIT OR Apache-2.0 | gh:dtolnay/thiserror |
| `thiserror-impl` | 1.0.69 | MIT OR Apache-2.0 | gh:dtolnay/thiserror |
| `thiserror-impl` | 2.0.20 | MIT OR Apache-2.0 | gh:dtolnay/thiserror |
| `tokenizers` | 0.21.4 | Apache-2.0 | gh:huggingface/tokenizers |
| `tokio` | 1.53.1 | MIT | gh:tokio-rs/tokio |
| `tokio-macros` | 2.7.2 | MIT | gh:tokio-rs/tokio |
| `toml` | 0.8.23 | MIT OR Apache-2.0 | gh:toml-rs/toml |
| `toml_datetime` | 0.6.11 | MIT OR Apache-2.0 | gh:toml-rs/toml |
| `toml_edit` | 0.22.27 | MIT OR Apache-2.0 | gh:toml-rs/toml |
| `toml_write` | 0.1.2 | MIT OR Apache-2.0 | gh:toml-rs/toml |
| `unicode-ident` | 1.0.26 | (MIT OR Apache-2.0) AND Unicode-3.0 | gh:dtolnay/unicode-ident |
| `unicode-normalization-alignments` | 0.1.12 | MIT/Apache-2.0 | gh:n1t0/unicode-normalization |
| `unicode-segmentation` | 1.13.3 | MIT OR Apache-2.0 | gh:unicode-rs/unicode-segmentation |
| `unicode_categories` | 0.1.1 | MIT OR Apache-2.0 | gh:swgillespie/unicode-categories |
| `version_check` | 0.9.5 | MIT/Apache-2.0 | gh:SergioBenitez/version_check |
| `wasi` | 0.11.1+wasi-snapshot-preview1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | gh:bytecodealliance/wasi |
| `wasip2` | 1.0.4+wasi-0.2.12 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | gh:bytecodealliance/wasi-rs |
| `windows-link` | 0.2.1 | MIT OR Apache-2.0 | gh:microsoft/windows-rs |
| `windows-sys` | 0.61.2 | MIT OR Apache-2.0 | gh:microsoft/windows-rs |
| `winnow` | 0.7.15 | MIT | gh:winnow-rs/winnow |
| `wit-bindgen` | 0.57.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | gh:bytecodealliance/wit-bindgen |
| `zerocopy` | 0.8.57 | BSD-2-Clause OR Apache-2.0 OR MIT | gh:google/zerocopy |
| `zerocopy-derive` | 0.8.57 | BSD-2-Clause OR Apache-2.0 OR MIT | gh:google/zerocopy |
| `zlib-rs` | 0.6.8 | Zlib | gh:trifectatechfoundation/zlib-rs |
| `zmij` | 1.0.23 | MIT | gh:dtolnay/zmij |
| `zune-core` | 0.5.3 | MIT OR Apache-2.0 OR Zlib | gh:etemesi254/zune-image |
| `zune-jpeg` | 0.5.15 | MIT OR Apache-2.0 OR Zlib | gh:etemesi254/zune-image/tree/dev/crates/zune-jpeg |
