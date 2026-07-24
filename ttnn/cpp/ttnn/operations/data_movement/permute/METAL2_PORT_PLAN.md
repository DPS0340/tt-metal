# Port Plan — permute (`ttnn/cpp/ttnn/operations/data_movement/permute`)

Port plan for `permute`, ported from the TTNN `ProgramDescriptorFactoryConcept` (`create_descriptor` → `ProgramDescriptor`) to Metal 2.0 (`MetalV2FactoryConcept` → `create_program_artifacts` → `ProgramArtifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

> **Atomic unit = one ProgramFactory.** `PermuteDeviceOperation` holds five factories in `program_factory_t`. They are ported one at a time; a `program_factory_t` variant is valid with factories on mixed concepts (framework dispatches per-factory). This plan inventories all five, then plans the port pass-by-pass.

## Porting order (passes)

| Pass | Factory | File | Kernels | Donor kernels? | Status |
|---|---|---|---|---|---|
| 1 | `MultiCoreRowInvariant` | `permute_rm_program_factory.cpp` | reader + writer (own) | no | planned |
| 1 | `MultiCoreBlockedGeneric` | `permute_rm_program_factory.cpp` | reader + writer + compute (own) | no | planned |
| 2 | `MultiCoreTiledGeneric` | `permute_tiled_program_factory.cpp` | reader + writer + compute (own) | no | deferred |
| 3 | `MultiCoreTileInvariant` | `permute_tiled_program_factory.cpp` | reader (own) + writer (donor) + compute (donor, swap-hw) | **yes** | deferred |
| 3 | `MultiCoreTileRowInvariant` | `permute_tiled_program_factory.cpp` | reader (donor) + writer (own) + compute (donor, swap-hw) | **yes** | deferred |

Rationale: the RM file (pass 1) has two self-contained factories with no donor kernels — the cleanest starting unit and the entire row-major path. `MultiCoreTiledGeneric` (pass 2) is also donor-free. The two remaining tiled factories (pass 3) instantiate three donor kernels by file path (`writer_unary_interleaved_start_id.cpp`, `transpose_wh.cpp`, `reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp`) and require the shared-kernel fork-vs-in-place decision; grouped last so the donor rewrite is a single coordinated step.

---

## Legacy Inventory

### Legacy factory shape
- **Concept:** `ProgramDescriptorFactoryConcept` — each factory is `static ProgramDescriptor create_descriptor(...)` ([permute_device_operation.hpp:36-75](device/permute_device_operation.hpp#L36-L75)).
- **Variants:** five factories in `program_factory_t` (`MultiCoreRowInvariant`, `MultiCoreBlockedGeneric`, `MultiCoreTileInvariant`, `MultiCoreTileRowInvariant`, `MultiCoreTiledGeneric`); `select_program_factory` chooses by layout + which dims move ([permute_rm_program_factory.cpp:19-43](device/permute_rm_program_factory.cpp#L19-L43)).
- **Custom `compute_program_hash`:** none — no override in the op (default reflection-based hash). No deletion required.

*(Target Metal 2.0 concept `MetalV2FactoryConcept`, chosen by the audit — carried forward in [TTNN ProgramFactory](#ttnn-programfactory).)*

Bindings shared by **all** factories (per audit, all **Case 1** via `TensorAccessor`):
- `input_tensor` (`src_buffer`) — delivered today via the `Buffer*`-binding form (`emplace_runtime_args(core, {src_buffer, …})`), kernel builds `TensorAccessor(src_args, src_addr)`. → `TensorParameter INPUT`.
- `output_tensor` (`dst_buffer`) — symmetric. → `TensorParameter OUTPUT`.

No `->address()` anywhere (clean bases, no host-folded offset); no semaphores in any factory; every `TensorAccessor` construction is the 2-arg form.

### Variant: MultiCoreRowInvariant  *(RM, last dim unchanged)*

#### Kernels
| unique_id | source | core_ranges | CTAs (named) | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_permute_interleaved_rm_row_invariant.cpp` | all_cores | `N`, `page_size`(=input rm page), `num_rows` + `TensorAccessorArgs(src)` | none | `{src_buffer, start_row, end_row}` | `ReaderConfigDescriptor{}` (reader default) |
| writer | `device/kernels/dataflow/writer_permute_interleaved_rm_row_invariant.cpp` | all_cores | `N`, `page_size`(=output rm page), `num_rows` + `TensorAccessorArgs(dst)` | none | `{dst_buffer, start_row, end_row}` + trailing `[input_shape(N), perm(N), dest_strides(N)]` | `WriterConfigDescriptor{}` (writer default) |

- Reader kernel uses named CTAs `N`/`num_rows` but only `page_size` is read in the body (N, num_rows dead — audit "harmless"). Constructs `DataflowBuffer dfb(tt::CBIndex::c_0)` (magic index), PRODUCER of c_0. `TensorAccessor s0(src_args, src_addr)`.
- Writer reads `N` (used as loop bound), `page_size` (used), `num_rows` (dead). Reads varargs block `input_shape[N]/perm[N]/dest_strides[N]` in loop `for i=3..N+3` at offsets `i, i+N, i+2N`. `DataflowBuffer dfb(tt::CBIndex::c_0)`, CONSUMER. Dead local `curr_addr = dst_addr` ([writer_...rm_row_invariant.cpp:34](device/kernels/dataflow/writer_permute_interleaved_rm_row_invariant.cpp#L34)) — removed by port (its only input `dst_addr` disappears with the TensorBinding).

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| c_0 | `2 * input_rm_page_size` | all_cores | `datatype_to_dataformat_converter(input.dtype())` | `input_rm_page_size` | (unset) |

#### Work split
`split_work_to_cores(compute_with_storage_grid_size, num_rows)` — one WorkUnit over `all_cores`; per-core `start_row/end_row`. Groups 1/2 differ only in per-core row count → runtime, not CTA (no per-group CTA multiplicity).

### Variant: MultiCoreBlockedGeneric  *(RM, last dim moved — blocked transpose)*

#### Kernels
| unique_id | source | core_ranges | CTAs (named) | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|---|
| reader | `.../dataflow/reader_permute_interleaved_rm_blocked_generic.cpp` | all_cores | `N, page_size, num_rows, x_dim, num_blocks_total, x_blocks, w_blocks, x_block_size, w_block_size, element_size, input_tensor_page_size` + `TensorAccessorArgs(src)` | none | `{src_buffer, start_block, end_block}` + `[input_shape(N), src_strides(N)]` | reader default |
| writer | `.../dataflow/writer_permute_interleaved_rm_blocked_generic.cpp` | all_cores | `N, output_page_size, num_rows, X, X_stride, x_dim, W_stride, input_page_size, element_size, num_blocks_total, x_blocks, w_blocks, x_block_size, w_block_size, W, output_tensor_page_size` + `TensorAccessorArgs(dst)` | none | `{dst_buffer, start_block, end_block}` + `[input_shape(N), perm(N), dest_strides(N)]` | writer default |
| compute | `.../compute/transpose_xw_rm_single_tile_size.cpp` | all_cores | `x_block_size, w_block_size` | `{x_block_size, w_block_size}` (DEAD — kernel reads named; positional reads are commented out) | `{num_blocks_per_core, 0u, 0u}` (reads slot 0 only; two `0u` DEAD) | `ComputeConfigDescriptor{.fp32_dest_acc_en = (out fmt ∈ {FP32,Int32,UInt32})}` |

- Reader: `DataflowBuffer dfb(tt::CBIndex::c_0)` (per-iteration construct), PRODUCER c_0. Varargs 2N.
- Compute: `cb_in=c_0`, `cb_tilize=c_1`, `cb_out=c_2` (magic indices, passed to `compute_kernel_hw_startup`/`unary_op_init_common`/`compute_kernel_lib::tilize<1,cb_in,cb_tilize,…>` and `DataflowBuffer dfb_tilize_exp(c_1)`/`dfb_out_exp(c_2)`). Consumes c_0 (via tilize helper), self-loops c_1 (tilize produces + `wait_front`/`pop_front` consumes), produces c_2. `num_blocks = get_arg_val<uint32_t>(0)`.
- Writer: `DataflowBuffer dfb_in(c_2)` (`dfb_id_in = c_2`), CONSUMER c_2. Uses `swap_elements` from common.hpp. Varargs 3N.

#### CBs
| index | role | total_size | data_format | page_size | tile |
|---|---|---|---|---|---|
| c_0 (src0) | input | `2 * input_cb_page_size * x_block_size` | input dtype fmt | `input_cb_page_size` (= `w_block_size * elem`) | (unset) |
| c_2 (src1) | output staging | `2 * output_cb_page_size * w_block_size` | input dtype fmt | `output_cb_page_size` (= `x_block_size * elem`) | (unset) |
| c_1 (src2) | tilize intermediate (**self-loop**) | `2 * x_block_size * w_block_size * elem` | input dtype fmt | `x_block_size * w_block_size * elem` | (unset) |

#### Work split
`split_work_to_cores(grid, num_blocks_total)` where `num_blocks_total = (num_rows/X) * x_blocks * w_blocks`. One WorkUnit over `all_cores`.

### Variant: MultiCoreTiledGeneric  *(tiled, both tile dims moved)* — pass 2

#### Kernels
| unique_id | source | CTAs (named) | RTAs | config |
|---|---|---|---|---|
| reader | `.../dataflow/reader_permute_interleaved_tiled_generic.cpp` | 28 named (`rank, page_size, element_size, tile_*, face_*, x_dim_index_in_input, X, W, H, X_p, W_p, H_p, H_t, W_t, final_tile_real_w, final_tile_real_faces_w, xw_blocks, x_blocks, w_blocks, num_writes, padding_val_packed, needs_x_padding, needs_y_padding, rows_per_x, misalignment, read_alignment`) + `TensorAccessorArgs(src)` | `{src_buffer, start_block, end_block}` + `[input_shape(rank), dims(rank)]` (2·rank varargs) | reader default |
| writer | `.../dataflow/writer_permute_interleaved_tiled_generic.cpp` | 22 named + `TensorAccessorArgs(dst)` | `{dst_buffer, start_block, end_block, start_tile_padding, end_tile_padding}` + `[input_shape(rank), dims(rank)]` (2·rank varargs) | writer default |
| compute | `.../compute/transpose_xw_tiled.cpp` | none | `{start_block, end_block}` | `ComputeConfigDescriptor{.fp32_dest_acc_en=(fmt∈{FP32,Int32,UInt32})}` (no `unpack_to_dest_mode`) |

#### CBs
| index | role | page_size | tile |
|---|---|---|---|
| c_0 (src0) | input | `input_page_size = tile_size(out) + misalignment` | (unset) |
| c_1 (src1) | tilize intermediate (**self-loop**, compute) | same | (unset) |
| c_2 (src2) | output staging | same | (unset) |
| c_3 (padding) | pad tiles, only if `needs_y_padding` | `face_shape[1] * elem` | (unset) |

Endpoints: c_0 reader P / compute C; c_1 compute self-loop; c_2 compute P / writer C; c_3 reader P / writer C. Work split `split_work_to_cores(grid, xw_blocks)`.

### Variant: MultiCoreTileInvariant  *(tiled, tile dims stay in last two — identity or WH swap)* — pass 3

- Configs: **non-swap** (reader→writer) and **swap-hw** (reader→compute→writer).
- **Donor kernels:** writer = `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (cross-family); compute (swap only) = `data_movement/transpose/device/kernels/compute/transpose_wh.cpp` (in-family).
- Reader (own) `reader_permute_interleaved_tiled_invariant.cpp`: named CTAs `rank, page_size, num_tiles` + `TensorAccessorArgs(src)`; RTAs `{src, start_tile, end_tile}` + `[output_shape(rank), inv_perm(rank), input_tile_strides(rank)]` (3·rank varargs); PRODUCER c_0.
- Writer (donor unary): positional CTA `{output_cb_index}` (magic CB index → `DFBBinding`) + `TensorAccessorArgs(dst)`; RTAs `{dst, num_tiles_per_core, start_tile}`; CONSUMER of output CB.
- Compute (donor transpose_wh, swap only): empty CTAs; RTA `{num_tiles_per_core}`; `ComputeConfigDescriptor{.fp32_dest_acc_en, .unpack_to_dest_mode=[Default×NUM_CB, src0→UnpackToDestFp32 if Float32]}`. CONSUMER c_0, PRODUCER c_16.
- CBs: c_0 (input); c_16 (src1, output) only when swap_hw. `output_cb_index = swap_hw ? c_16 : c_0`. Work split `split_work_to_cores(grid, num_tiles)`.

### Variant: MultiCoreTileRowInvariant  *(tiled, one tile dim moved out)* — pass 3

- Configs: **swap_hw × needs_padding**.
- **Donor kernels:** reader = `data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp` (in-family); compute (swap only) = `transpose_wh.cpp` (in-family).
- Reader (donor): 9 named CTAs (`num_writes, padding_val_packed, needs_padding, swap_hw, H, W, accumulated_outer_dims, tile_height, tile_width`) + `TensorAccessorArgs(src)`; RTA `{src, num_tiles_per_core, start_tile}`; PRODUCER c_0 (and c_1 padding).
- Writer (own) `writer_permute_interleaved_tiled_row_invariant.cpp`: 12 named CTAs + `TensorAccessorArgs(dst)`; RTAs `{dst, start_tile, end_tile, start_tile_padding, end_tile_padding}` + `[input_shape(rank), dims(rank)]` (2·rank varargs); CONSUMER c_0/c_16 and c_1.
- Compute (donor transpose_wh, swap only): as TileInvariant.
- CBs: c_0 (input); c_1 (padding) if `needs_padding` (reader P / writer C); c_16 (src2, output) if swap_hw (compute P / writer C). `output_cb_index = swap_hw ? c_16 : c_0`. Two work splits (`num_tiles` and `padded_num_tensor_tiles`), `all_cores` = the larger grid.

### Cross-op kernels
Three donor kernel sources outside the op directory (all instantiated by file path; all Device-2.0 compliant per audit). Each is a [Caution: shared dataflow kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md) case, reported under "Open items for downstream":
- `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — broadly shared (cross-family). Used by `MultiCoreTileInvariant`.
- `data_movement/transpose/device/kernels/compute/transpose_wh.cpp` — in-family (transpose). Used by `MultiCoreTileInvariant` + `MultiCoreTileRowInvariant` (swap-hw).
- `data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp` — in-family (transpose). Used by `MultiCoreTileRowInvariant`.

All three are pass-3 concerns.

### Flags
- No unreferenced kernel files (all 10 owned kernels referenced by a factory).
- Misc dead code (audit, non-gating): compute dead RTA slots `{_,0u,0u}` in BlockedGeneric; dead `curr_addr` in RM row-invariant writer + tiled-invariant reader; unused CTAs `N`/`num_rows` in RM row-invariant reader. The port drops only what the binding conversion forces (e.g. `curr_addr = dst_addr` when `dst_addr` disappears); it does not otherwise touch dead code.

---

## TTNN ProgramFactory

*Planning step — carries forward the audit's decision.*

- **Concept (inherited from audit):** `MetalV2FactoryConcept` (no op-owned tensors; single-program).
- **Custom `compute_program_hash`:** none — nothing to delete.
- **Implementation notes:** device-op class edits — the port changes each ported factory's declaration in `permute_device_operation.hpp` from `static ProgramDescriptor create_descriptor(...)` to `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`. The `program_factory_t` variant keeps unported factories on `create_descriptor` (mixed-concept variant is valid). No pybind `create_program_descriptor` exists (nanobind binds via `bind_function<"permute">`, [permute_nanobind.cpp:35](permute_nanobind.cpp#L35)) → no pybind removal. No custom-hash deletion. No pybind-hook-only factory parameter.

---

## Planned Spec Shape

### Pass 1 — MultiCoreRowInvariant

- **KernelSpecs:** `READER` (reader_...rm_row_invariant.cpp), `WRITER` (writer_...rm_row_invariant.cpp). 1:1 with legacy.
- **DataflowBufferSpecs:** `SRC_CB` (c_0): `entry_size = input_rm_page_size`, `num_entries = 2`, `data_format_metadata = input dtype fmt`, no tile metadata.
- **TensorParameters:** `INPUT` (input_tensor.tensor_spec()), `OUTPUT` (output_tensor.tensor_spec()).
- **SemaphoreSpecs:** none.
- **WorkUnitSpecs:** one — `{READER, WRITER}` over `all_cores`.
- **Bindings:** READER → SRC_CB PRODUCER + INPUT (`tensor::input`); WRITER → SRC_CB CONSUMER + OUTPUT (`tensor::output`).

### Pass 1 — MultiCoreBlockedGeneric

- **KernelSpecs:** `READER`, `WRITER`, `COMPUTE`. 1:1 with legacy.
- **DataflowBufferSpecs:** `SRC_CB` (c_0), `OUT_CB` (c_2), `TILIZE_CB` (c_1). entry_size/num_entries copied from legacy CB `page_size`/`(total_size/page_size)`.
- **TensorParameters:** `INPUT`, `OUTPUT`.
- **WorkUnitSpecs:** one — `{READER, WRITER, COMPUTE}` over `all_cores`.
- **Bindings:** READER → SRC_CB PRODUCER + INPUT; COMPUTE → SRC_CB CONSUMER, TILIZE_CB **PRODUCER + CONSUMER (self-loop)**, OUT_CB PRODUCER; WRITER → OUT_CB CONSUMER + OUTPUT.

### Pass 2 / Pass 3 — deferred

Detailed spec shape for `MultiCoreTiledGeneric` (pass 2) and `MultiCoreTileInvariant` / `MultiCoreTileRowInvariant` (pass 3) is written when those passes begin. Structural preview in the Legacy Inventory above.

## Preserved Multiplicity

**None — no work-split multiplicity in legacy.** Every factory uses a single `split_work_to_cores` and emits **one** `KernelDescriptor` per source over `all_cores`; core groups 1/2 differ only in per-core scalar counts (runtime), not per-group CTAs. So each factory maps to one WorkUnit over `all_cores`; no per-group `KernelSpec` duplication. (TileRowInvariant/TiledGeneric have a *second* padding work-split, but it only feeds per-core padding scalars — still runtime, not CTA multiplicity.)

## Dropped Plumbing

### Pass 1 — MultiCoreRowInvariant
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slot 0 (`src_buffer`) | `Buffer*` binding → `src_addr` | `TensorBinding(INPUT)`; kernel `TensorAccessor(tensor::input)` |
| writer RTA slot 0 (`dst_buffer`) | `Buffer*` binding → `dst_addr` | `TensorBinding(OUTPUT)`; kernel `TensorAccessor(tensor::output)` |
| reader/writer `TensorAccessorArgs(*buf).append_to(cta)` | positional CTA plumbing | binding supplies accessor args; `TensorAccessorArgs<0>()` kernel line drops |
| reader/writer `DataflowBuffer dfb(tt::CBIndex::c_0)` | magic CB index | `DFBBinding` → `DataflowBuffer dfb(dfb::src_cb)` |
| writer RTA slots 3..3+3N | positional shape/perm/stride arrays (count = N, a CTA) | **runtime varargs** (`num_runtime_varargs = 3N`); kernel `get_vararg(i)` |
| writer local `curr_addr = dst_addr` | dead local off `dst_addr` | removed (forced — `dst_addr` gone) |
| all positional CTA reads | `get_named_compile_time_arg_val` / `get_arg_val` | `get_arg(args::name)` (named throughout) |

### Pass 1 — MultiCoreBlockedGeneric
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader/writer buffer RTA slot 0 | `Buffer*` binding | `TensorBinding(INPUT)` / `TensorBinding(OUTPUT)` |
| reader/writer `TensorAccessorArgs(*buf).append_to` | positional CTA plumbing | binding |
| reader `DataflowBuffer dfb(c_0)`; compute `c_0/c_1/c_2` magic indices; writer `dfb_in(c_2)` | magic CB indices | `DFBBinding` → `dfb::src_cb` / `dfb::tilize` / `dfb::out` |
| compute positional CTA `{x_block_size, w_block_size}` | dead positional (kernel reads named) | dropped (named CTAs only) |
| compute RTA slots 1,2 (`0u, 0u`) | dead padding slots | dropped (named `num_blocks` only) |
| reader RTA 3..3+2N; writer RTA 3..3+3N | positional shape/stride/perm arrays (count = N CTA) | runtime varargs (2N / 3N) |

## Applied Patterns
- **[Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md):** `TILIZE_CB` (c_1) on `MultiCoreBlockedGeneric` COMPUTE — bind PRODUCER **and** CONSUMER (single toucher: tilize produces, transpose consumes, both inside the one compute kernel). Same pattern will apply to `MultiCoreTiledGeneric` c_1 (pass 2).
- **Pass DFB handles directly to LLKs / kernel-lib helpers:** BlockedGeneric compute passes `dfb::src_cb`/`dfb::tilize`/`dfb::out` into `compute_kernel_hw_startup`, `unary_op_init_common`, and `compute_kernel_lib::tilize<1, dfb::src_cb, dfb::tilize, …>` (NTTP position — `DFBAccessor::operator uint32_t()` is constexpr).
- **Runtime varargs (kept, not promoted):** the rank-length shape/perm/stride arrays in the RM reader/writer kernels are genuine indexed-collection elements (loop count = `N`, a CTA) → `num_runtime_varargs`. Per-core scalars (`start_row/end_row`, `start_block/end_block`, `num_blocks`) stay named RTAs. Reported per recipe.

## Deferred / Flagged
- **unpack_modes (BlockedGeneric compute):** legacy uses `ComputeConfigDescriptor{.fp32_dest_acc_en=…}` with **no** `unpack_to_dest_mode` (Style B). Under Metal 2.0, when `enable_32_bit_dest = true` and a consumed DFB is `Float32`, an explicit `unpack_modes` entry is required. Plan: build `ComputeGen1Config{.enable_32_bit_dest = fp32_dest_acc_en}`; when `cb_data_format == Float32`, add `unpack_modes = {{SRC_CB, UnpackToSrc}, {TILIZE_CB, UnpackToSrc}}` (compute consumes both; legacy default → `UnpackToSrc`). Int32/UInt32 not required (issue #49936). Verify the exact set of consumed FP32 DFBs at construction.
- **hw_config (DM):** all reader/writer configs are `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` (defaults) → arch-agnostic `create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)`.
- **hw_config (compute, BlockedGeneric):** Style B → `ComputeGen1Config`, `enable_32_bit_dest = fp32_dest_acc_en`; all other fields default (match legacy `ComputeConfigDescriptor` defaults). `bfp_pack_precision_mode` left default (legacy did not set `bfp8_pack_precise`).
- **Pass 2/3 detail** (tiled compute configs with `unpack_to_dest_mode`, donor-kernel fork decisions, padding-CB endpoints) is planned when those passes begin.
