# Metal 2.0 Port Report — permute (`ttnn/cpp/ttnn/operations/data_movement/permute`)

## Outcome

**PORTED (partial, by design)** — pass 1 of 3: the two **row-major** factories are on Metal 2.0 (`MetalV2FactoryConcept`):
- `MultiCoreRowInvariant` — `create_program_artifacts`
- `MultiCoreBlockedGeneric` — `create_program_artifacts`

The three **tiled** factories (`MultiCoreTileInvariant`, `MultiCoreTileRowInvariant`, `MultiCoreTiledGeneric`) remain on the legacy `ProgramDescriptorFactoryConcept` (`create_descriptor`) and are enumerated for later passes in `METAL2_PORT_PLAN.md`. The `program_factory_t` variant is mixed-concept; the framework dispatches per-factory, so the op builds and runs with both RM paths on Metal 2.0 and the tiled paths on legacy. Scope of this session (pass 1 only) was confirmed with the invoker.

### Verification (blackhole)
- **Build:** `./build_metal.sh --build-tests` — clean, no errors/warnings on permute.
- **Unit — `tests/ttnn/unit_tests/operations/data_movement/test_permute.py`:** **1593 passed, 1 skipped** (`SAFE_PYTEST_RESULT: PASS`). Exercises all five `select_program_factory` paths (both ported RM + the three still-legacy tiled).
- **Nightly — `test_universal_input_tm_permute.py`:** **86 passed**, identical to the pre-port baseline (86 passed) captured on this branch before any changes — **no regression**.
- **Anti-pattern self-audit:** clean — no `buffer()->address()`, no CB-index CTAs, no `TensorAccessorArgs<N>()`, no `.id` extraction, no `allow_instance_multi_binding`, all CTAs named, varargs only for the genuine rank-length collections, every `hw_config` reproduces the legacy resolved values.

## Provenance

- **Recipe docs (this port):** `9ebb69d90cb 2026-07-24 docs(metal_2.0): fix dangling capitulation report target + add Outcome marker` — on branch `akertesz/op-porting-recipe`. The recipe docs are **not** present on the port branch's HEAD, so `git log -1 … -- docs/.../metal_2.0/` prints nothing here; the version is pinned to the `akertesz/op-porting-recipe` commit the port was run against (recorded above) instead.
- **Audit docs (inherited):** `2a53d817976 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for both ported factories — as the audit chose. No deviation.

### Device-op-class edits
- **Custom `compute_program_hash` deleted:** none — the op never had one (default reflection-based hash).
- **Pybind entry points removed:** none — the nanobind layer binds via `ttnn::bind_function<"permute">`; there was no `create_program_descriptor` pybind hook to remove.
- **Header change (forced, sanctioned):** `permute_device_operation.hpp` — the two ported factory structs' declarations changed from `static ProgramDescriptor create_descriptor(...)` to `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`; added `#include "ttnn/metal_v2_artifacts.hpp"`. The other three factory structs are untouched.

### Open items
- **Tiled factories (pass 2/3):** three factories remain on legacy. Pass 3 requires porting three **donor kernels** (see Open items for downstream).
- **Relaxation candidates:** none applied. `TensorParameter`s kept strict (default). Not investigated for relaxation — out of scope for the port.

## Handoff points

None for pass 1 (no capitulation, no boundary-rule assumption violations, no kernel-lib gaps, no framework gaps, no removed pybind surface). The RM factories are entirely self-contained within the op directory.

## Successes

- **Self-loop DFB (`port_patterns.md` — Self-loop DFB binding):** `MultiCoreBlockedGeneric`'s tilize CB (`cb_tilize`, legacy c_1) is produced by the tilize helper and consumed by the transpose, both inside the one compute kernel. The pattern's prescription (bind the same DFB PRODUCER + CONSUMER on the one kernel, shared accessor name) applied directly — `permute_rm_program_factory.cpp` compute `dfb_bindings`.
- **`unpack_modes` required-entry rule (`migration_guide.md` — DataflowBufferSpec / compute config):** the doc's warning fired exactly as described — legacy `ComputeConfigDescriptor` set no `unpack_to_dest_mode`, but Metal 2.0 requires an explicit entry for each Float32 DFB the compute kernel consumes when `enable_32_bit_dest = true`. Caught at planning time (not left to a validator failure); mapped legacy `Default` → `UnpackMode::UnpackToSrc` for the two consumed FP32 DFBs (`cb_in`, `cb_tilize`). Confined to the Float32 case (Int32/UInt32 deferred per #49936).
- **Runtime varargs rule (`port_patterns.md` — Avoid varargs):** the rank-length shape/perm/stride arrays are genuine indexed-collection elements (loop count = `N`, a CTA), so they stayed varargs while the per-core scalars (`start_row/end_row`, `start_block/end_block`, `num_blocks`) became named RTAs. The distinction was unambiguous following the caution.

## Friction

### Gaps
- *(none observed in pass 1 — update if the build/tests surface any)*

### Confusion
- **KernelSpec designated-initializer field order.** `advanced_options` sits *after* `hw_config` in the `KernelSpec` struct, so the varargs-carrying kernels must list `.hw_config` before `.advanced_options` (C++ designated initializers must follow declaration order). Easy to get backwards when mentally grouping "schema-ish" fields together. Minor; noting in case the recipe wants a field-order reminder near the varargs guidance.

## Open items for downstream

- **Dead named CTAs dropped (RM blocked generic).** Legacy emitted `input_tensor_page_size` (reader) and `output_tensor_page_size` (writer), both from `buffer()->aligned_page_size()`, that **neither kernel reads**. Dropped from the Metal 2.0 CTA tables (a Metal 2.0 named CTA should correspond to a kernel `args::` read; emitting a dead one is clutter). Zero functional change. Flagging for the ops team in case the intent was to wire them somewhere.
- **Dead commented-out positional CTA reads removed (compute kernel).** `transpose_xw_rm_single_tile_size.cpp` carried two commented-out `get_compile_time_arg_val(0/1)` lines (a pre-named-CTA leftover). Removed as part of the named-CTA conversion — they were commented-out code referencing a removed API, not explanatory comments.
- **Dead compute RTA slots dropped.** Legacy `MultiCoreBlockedGeneric` compute emitted `{num_blocks_per_core, 0u, 0u}` (kernel read slot 0 only). The two trailing `0u` "historical layout" slots are gone (named `num_blocks` only). Zero functional change. (Audit "Misc anomalies" flagged this; the port drops it as a natural consequence of named args, not a separate change.)
- **Tiled factories + donor kernels (pass 2/3).** `MultiCoreTiledGeneric` (pass 2, donor-free) and the two donor-using tiled factories (pass 3) remain. Pass 3 touches three cross-op donor kernels instantiated by file path — each needs the fork-vs-in-place decision (`port_patterns.md` — Modifying a shared dataflow kernel):
  - `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (broadly shared, cross-family) — used by `MultiCoreTileInvariant`.
  - `data_movement/transpose/device/kernels/compute/transpose_wh.cpp` (in-family) — used by `MultiCoreTileInvariant` + `MultiCoreTileRowInvariant` (swap-hw).
  - `data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp` (in-family) — used by `MultiCoreTileRowInvariant`.
  Neither has been touched in pass 1.
