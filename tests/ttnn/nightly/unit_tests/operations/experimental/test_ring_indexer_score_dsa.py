# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness of the ring-fused indexer_score op (ttnn.experimental.ring_indexer_score_dsa) on a ring of 4
(a (1, 4) mesh; the op is SP=4, so it runs on 4 devices). One op co-schedules the
ring_attention all-gather with the score; the reader gates each K band on only the SP shards it touches and
dual-sources its own slab from k_local. Checked vs the same per-SP DSA reference the two-op path uses, over
both K layouts, both head counts, and both link counts, plus the runtime knobs (bfp8_b K, multi-user cache,
straddle, kv_len, program-cache reuse, validate reject). Also covers the 2D SP×TP layout on a (2, 2) mesh under
FABRIC_2D (test_indexer_score_sptp_fused, at the bottom).

Run:  scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/test_ring_indexer_score_dsa.py
"""

import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score import (
    assert_indexer_match,
    glx_config,
    indexer_score_dsa_ref,
    _global_inputs,
    _per_sp_ref,
    _straddle_ref,
    _to_slab,
    QB_HISTORY,
    QB_SQ,
    QB_CASES,
    QB_IDS,
    ST_CHUNK,
    ST_CS,
    ST_T,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.ring4_ccl_helpers import (
    _open_ring4_ccl,
    _close_ring4_ccl,
    _persistent_buffer,
    _shard_k,
    RING,
    SP_AXIS,
    CHUNK_GLOBAL,
    T,
)

pytestmark = [
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only"),
    pytest.mark.skipif(ttnn.get_num_devices() < 4, reason="ring-of-4 needs 4 devices"),
]


def _fused_dev_inputs(submesh, q_g, w_g, k_host, *, k_dtype=ttnn.bfloat16):
    """Fused op inputs: SP-shard q/w (bf16) on dim 2, SP-shard k_local (the AG input), and a zero-seeded
    gathered buffer (AG fills remote bands; zeros prove the reader dual-sources the local band). k_dtype sets
    both k_local and k_gathered (the op requires them equal)."""
    shard = ttnn.ShardTensorToMesh(submesh, dim=2)
    q_dev = ttnn.from_torch(q_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
    w_dev = ttnn.from_torch(w_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
    k_local = _shard_k(submesh, k_host, dtype=k_dtype)  # [B,1,sll,D] per chip (the all-gather INPUT)
    k_gathered = _persistent_buffer(submesh, torch.zeros_like(k_host), dtype=k_dtype)  # [B,1,T,D] AG OUTPUT
    return q_dev, w_dev, k_local, k_gathered


def _run_fused(
    heads,
    *,
    block_cyclic,
    num_links=1,
    k_dtype=ttnn.bfloat16,
    topology=ttnn.Topology.Linear,
    fabric_config=ttnn.FabricConfig.FABRIC_2D,
):
    """Run the one fused op and check vs the per-SP reference. num_links only changes fabric routing, never
    the gathered result -> same reference."""
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl(fabric_config)
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
        k_host = _to_slab(k_nat, RING, CHUNK_GLOBAL) if block_cyclic else k_nat
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_host, k_dtype=k_dtype)

        bc_kwargs = dict(block_cyclic_sp_axis=SP_AXIS, block_cyclic_chunk_local=QB_SQ) if block_cyclic else {}
        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=topology,
            num_links=num_links,
            ag_sub_device_id=subdevice_id,
            program_config=glx_config(heads),
            **bc_kwargs,
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        ref = _per_sp_ref(q_g, k_nat, w_g, RING, QB_HISTORY)
        assert_indexer_match(out_t, ref, CHUNK_GLOBAL, T, check_neg=True)
        layout = "block_cyclic" if block_cyclic else "contiguous"
        logger.info(f"ring4 fused {layout} (heads={heads}): fused all-gather + dual-source score matched reference")
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic"])
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused(case_id, heads, block_cyclic):
    """Base fused path, num_links=2 (the production Blackhole link count)."""
    _run_fused(heads, block_cyclic=block_cyclic, num_links=2)


def test_indexer_score_ring4_fused_ring_topology():
    """Exercise the distinct wraparound neighbor/threshold path (Topology.Ring). This is the one case that
    stays on FABRIC_1D_RING rather than the 2D fabric the other tests use: the 2D torus equivalent needs an
    axis of extent > 2, which does not fit on a 4-device box."""
    _run_fused(
        16,
        block_cyclic=True,
        num_links=1,
        topology=ttnn.Topology.Ring,
        fabric_config=ttnn.FabricConfig.FABRIC_1D_RING,
    )


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic"])
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_bfp8_k(case_id, heads, block_cyclic):
    """Production dtype: bfloat8_b K (local shard + gathered buffer), q/w stay bf16. Same PCC floor."""
    _run_fused(heads, block_cyclic=block_cyclic, num_links=1, k_dtype=ttnn.bfloat8_b)


@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_production_shape(case_id, heads):
    """All production knobs at once (each covered alone elsewhere): block-cyclic + non-zero chunk_start +
    kv_len < T_alloc + num_links=2 + bfloat8_b K. Guards their interaction (bfp8 gathered buffer + kv_len tail
    mask + block-cyclic invP on the nl2 schedule), which the model always drives together."""
    chunk_start = CHUNK_GLOBAL  # a later prefill chunk (rank r attends to chunk_start + (r+1)*QB_SQ)
    kv_len = chunk_start + CHUNK_GLOBAL  # fullest rank's causal window == kv_len exactly (validate's tightest bound)
    t_alloc = 4 * CHUNK_GLOBAL  # over-allocate so kv_len < T_alloc (ring-divisible, tile-aligned)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, t_alloc, seed=42)
        k_bc = _to_slab(k_nat, RING, CHUNK_GLOBAL)  # block-cyclic physical layout the reader inverts
        # bfloat8_b K (the model's cache dtype) for both the local shard and the gathered buffer.
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_bc, k_dtype=ttnn.bfloat8_b)

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=2,
            ag_sub_device_id=subdevice_id,
            chunk_start_idx=chunk_start,
            kv_len=kv_len,
            block_cyclic_sp_axis=SP_AXIS,
            block_cyclic_chunk_local=QB_SQ,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        # Only [0, kv_len) is valid; each SP rank scores the valid key prefix at the non-zero chunk_start.
        ref = _per_sp_ref(q_g, k_nat[:, :, :kv_len, :], w_g, RING, chunk_start)
        assert_indexer_match(out_t[:, :, :, :kv_len], ref, CHUNK_GLOBAL, kv_len, check_neg=True)
        logger.info(
            f"ring4 fused production-shape (heads={heads}): block_cyclic+cs={chunk_start}+kv_len={kv_len}+nl2+bfp8 "
            f"matched reference"
        )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


def _run_fused_multiuser(heads, *, num_users, cache_batch_idx, num_links=1):
    """Multi-user indexed cache: k_local [num_users,1,sll,D] + gathered [num_users,1,T,D]; cache_batch_idx
    selects the slot in-kernel. Regression guard for the local-shard offset fix -- the reader must offset BOTH
    the remote and the local reads by the slot (pre-fix it read the local band from slot 0, mixing users; slots
    hold distinct K so any cache_batch_idx>0 fails PCC). An op capability the model itself doesn't use (it
    slices to batch-1 before the AG), guarded here for other callers."""
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        # Shared q/w scoring distinct per-user caches (distinct seed per slot -> a wrong-slot read changes the score).
        q_g, _, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
        k_multi = torch.cat([_global_inputs(heads, CHUNK_GLOBAL, T, seed=100 + u)[1] for u in range(num_users)], dim=0)
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_multi)

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=num_links,
            ag_sub_device_id=subdevice_id,
            cache_batch_idx=cache_batch_idx,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        ref = _per_sp_ref(q_g, k_multi[cache_batch_idx : cache_batch_idx + 1], w_g, RING, QB_HISTORY)
        assert_indexer_match(out_t, ref, CHUNK_GLOBAL, T, check_neg=True)
        logger.info(
            f"ring4 fused multi-user (heads={heads}, users={num_users}, slot={cache_batch_idx}): matched reference"
        )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


def test_indexer_score_ring4_fused_indexed_cache():
    """cache_batch_idx=1 (2nd user slot). One representative case (dsv32) -- the slot offset is head-independent."""
    _run_fused_multiuser(16, num_users=2, cache_batch_idx=1)


@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_straddle(case_id, heads):
    """Mid-slab straddle + block-cyclic rotation (rotated-prefill/multiturn): a non-slab-aligned chunk_start
    (704) makes the boundary chip's queries cross a slab boundary, so the causal diagonal jumps by
    (chunk_global - cl). Proves the band reorder + per-band gate + dual-source read compose with the straddled
    mask. Checked vs the per-SP rotated reference."""
    cl = ST_CHUNK // RING  # per-shard chunk / per-device query rows (block-cyclic SP-only)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, ST_CHUNK, ST_T, seed=42)
        k_bc = _to_slab(k_nat, RING, ST_CHUNK)  # block-cyclic physical layout the reader inverts

        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_bc)  # [1,1,ST_T/RING,D]

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=1,
            ag_sub_device_id=subdevice_id,
            chunk_start_idx=ST_CS,  # mid-slab (704 % cl != 0) -> rotation + straddle
            block_cyclic_sp_axis=SP_AXIS,
            block_cyclic_chunk_local=cl,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        ref = _straddle_ref(q_g, k_nat, w_g, RING, ST_CHUNK, ST_CS, ST_T)
        assert_indexer_match(out_t, ref, ST_CHUNK, ST_T, check_neg=True)
        logger.info(f"ring4 fused straddle (heads={heads}): rotated-prefill causal diagonal matched reference")
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


def test_indexer_score_ring4_fused_runtime_kv_len():
    """Padded cache: k allocated at T_alloc but only a kv_len prefix is valid; only cols [0, kv_len) are
    written. Confirms the AG gathers full T_alloc and the compute masks beyond kv_len (band_count spans full T,
    so no shard is left un-delivered). heads=16 representative -- kv_len masking is head-independent."""
    heads = 16
    kv_len = QB_HISTORY + CHUNK_GLOBAL  # valid written extent (28160 keys, 880 tiles)
    t_alloc = kv_len + CHUNK_GLOBAL  # over-allocate one more global chunk (30720, ring-divisible, tile-aligned)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, t_alloc, seed=42)
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_nat)  # [1,1,t_alloc/RING,D]

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=1,
            ag_sub_device_id=subdevice_id,
            chunk_start_idx=QB_HISTORY,  # rank r attends up to QB_HISTORY + (r+1)*QB_SQ; fullest = kv_len exactly
            kv_len=kv_len,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        # Only [0, kv_len) is valid; reference scores each rank against the valid key prefix (rest is stale tail).
        ref = _per_sp_ref(q_g, k_nat[:, :, :kv_len, :], w_g, RING, QB_HISTORY)
        assert_indexer_match(out_t[:, :, :, :kv_len], ref, CHUNK_GLOBAL, kv_len, check_neg=True)
        logger.info(
            f"ring4 fused runtime kv_len (kv_len={kv_len} of T_alloc={t_alloc}): valid prefix matched reference"
        )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


@pytest.mark.parametrize("k_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bfp8"])
def test_indexer_score_ring4_fused_program_cache_reuse(k_dtype):
    """Two dispatches, identical shapes but different chunk_start/kv_len on the SAME device (2nd is a cache
    hit). chunk_start/kv_len are hash-excluded, so override_runtime_arguments must re-apply them; if not, the
    2nd dispatch reuses the 1st's frozen offset -> wrong logits. Regression guard for the program-cache
    stale-scalar bug (every other test dispatches cold). Both bf16 and production bfp8_b K."""
    heads = 16  # dsv32; the scalar re-patch is head-independent, so one head count suffices (both dtypes kept)
    t_alloc = 4 * CHUNK_GLOBAL  # room for both chunks' causal windows (global block == CHUNK_GLOBAL)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, t_alloc, seed=42)
        k_bc = _to_slab(k_nat, RING, CHUNK_GLOBAL)  # block-cyclic physical layout
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_bc, k_dtype=k_dtype)

        def _score(chunk_start, kv_len):  # identical shapes each call -> 2nd is a program-cache hit
            out = ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                ccl_semaphores,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Linear,
                num_links=1,
                ag_sub_device_id=subdevice_id,
                chunk_start_idx=chunk_start,
                block_cyclic_sp_axis=SP_AXIS,
                block_cyclic_chunk_local=QB_SQ,
                kv_len=kv_len,
                program_config=glx_config(heads),
            )
            ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
            return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        # chunk@0 (cache miss/build) then chunk@CHUNK_GLOBAL (cache HIT -- must re-apply chunk_start + kv_len).
        out0 = _score(chunk_start=0, kv_len=CHUNK_GLOBAL)
        out1 = _score(chunk_start=CHUNK_GLOBAL, kv_len=2 * CHUNK_GLOBAL)
        ref0 = _per_sp_ref(q_g, k_nat[:, :, :CHUNK_GLOBAL, :], w_g, RING, 0)
        ref1 = _per_sp_ref(q_g, k_nat[:, :, : 2 * CHUNK_GLOBAL, :], w_g, RING, CHUNK_GLOBAL)
        assert_indexer_match(out0[:, :, :, :CHUNK_GLOBAL], ref0, CHUNK_GLOBAL, CHUNK_GLOBAL, check_neg=True)
        assert_indexer_match(out1[:, :, :, : 2 * CHUNK_GLOBAL], ref1, CHUNK_GLOBAL, 2 * CHUNK_GLOBAL, check_neg=True)
        logger.info(f"ring4 fused program-cache reuse (heads={heads}): 2nd chunk_start re-applied on cache hit")
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


def test_indexer_score_ring4_fused_rejects_head_streaming(expect_error):
    """The fused path requires all heads resident; a streaming config (0 < head_group_size < Hi) must be
    rejected at validate, not silently mis-scheduled. head-independent -> one representative case (dsv32)."""
    heads = 16  # dsv32; head_group_size=8 is a streaming config (0 < 8 < 16)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_nat)
        base = glx_config(heads)
        streaming_cfg = ttnn.IndexerScoreProgramConfig(
            q_chunk_size=base.q_chunk_size, k_chunk_size=base.k_chunk_size, head_group_size=heads // 2
        )
        with expect_error(RuntimeError, "head_group_size must be 0 or Hi"):
            ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                ccl_semaphores,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Linear,
                num_links=1,
                ag_sub_device_id=subdevice_id,
                program_config=streaming_cfg,
            )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


# ==== 2D SP×TP on a (2, 2) mesh: sp=2 ring (cluster_axis) × tp=2 query seq sub-shard (seq_subshard_axis) =====
# K stays SP-sharded + TP-replicated so the AG is unchanged; TP sub-shards the query rows and seq_subshard_axis
# threads each device's tp sub-offset into the causal score. Fused analogue of the classic-path
# test_indexer_score.py::test_indexer_score_sp2_tp2_seq_subshard_rotated. Runs under FABRIC_2D like the ring-of-4
# tests above, here exercising the fused AG + dual-source score over 2D routing on the second mesh axis (TP).
# Opens the (2, 2) mesh with its own opener (ring4_ccl_helpers is fixed to the (1, 4) ring-of-4 layout).
SP2 = 2  # sequence-parallel ranks == ring size (cluster_axis extent)
TP2 = 2  # tensor-parallel ranks the QUERY sequence is ALSO sub-sharded over (seq_subshard_axis extent)
SP2_AXIS = 0  # mesh rows == SP ring (cluster_axis / block_cyclic_sp_axis)
TP2_AXIS = 1  # mesh cols == TP seq sub-shard (seq_subshard_axis)
CHUNK_SPTP = SP2 * QB_SQ  # 1280 global chunk; per-SP-shard chunk_local = QB_SQ (640), per-device Sq = 320
T_SPTP = QB_HISTORY + CHUNK_SPTP  # 26880 keys


def _open_ccl(mesh_shape, fabric_config=ttnn.FabricConfig.FABRIC_2D):
    """Open `mesh_shape` directly, load a worker sub-device, make 2 ccl semaphores (the two ring directions).
    Same recipe as ring4_ccl_helpers._open_ring4_ccl but parameterized by mesh shape and fabric config; the SP×TP
    test opens a (2,2) mesh under FABRIC_2D (the default here). Returns
    (mesh, ccl_semaphores, worker_sub_device_id, stall_group)."""
    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))
        grid = mesh.compute_with_storage_grid_size()
        ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        worker_sub_device = ttnn.SubDevice([ccl_crs])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        stall_group = [worker_sub_device_id]
        mgr = mesh.create_sub_device_manager([worker_sub_device], 0)
        mesh.load_sub_device_manager(mgr)
        mesh.set_sub_device_stall_group(stall_group)
        ccl_semaphores = [ttnn.create_global_semaphore(mesh, ccl_crs, 0) for _ in range(2)]
        return mesh, ccl_semaphores, worker_sub_device_id, stall_group
    except Exception:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise


def _close_ccl(mesh):
    try:
        try:
            mesh.reset_sub_device_stall_group()
            mesh.clear_loaded_sub_device_manager()
        finally:
            ttnn.close_mesh_device(mesh)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _per_sp_tp_ref(q_g, k_g, w_g, sp, tp, history, sq_sp):
    """Reference for a 2D SP×TP seq sub-shard, in row-major device order (SP outer). Device (r, t) owns query
    rows [r*sq_sp + t*sq_dev, r*sq_sp + (t+1)*sq_dev) and scores from causal start history + that row base."""
    sq_dev = sq_sp // tp
    refs = []
    for r in range(sp):
        for t in range(tp):
            g0 = r * sq_sp + t * sq_dev
            sl = slice(g0, g0 + sq_dev)
            refs.append(indexer_score_dsa_ref(q_g[:, :, sl, :], k_g, w_g[:, :, sl, :], history + g0))
    return torch.cat(refs, dim=2)


@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_sptp_fused(case_id, heads):
    """2D SP×TP fused op on a (2,2) mesh under FABRIC_2D: sp=2 ring × tp=2 query seq sub-shard. Block-cyclic K
    (SP-sharded + TP-replicated, so the AG is unchanged); each device's causal diagonal starts at
    history + sp_rank*Sq_sp + tp_rank*Sq_dev -- proving the fused path threads the TP sub-offset into the score.
    Topology.Linear (the (2,2) SP axis is a ring of 2, no torus wraparound). Slab-aligned (no straddle); checked
    vs the per-(sp,tp) reference."""
    chunk_local = CHUNK_SPTP // SP2  # per-SP-shard chunk == QB_SQ (640); per-device query rows = 320
    chunk_start = QB_HISTORY  # slab-aligned (QB_HISTORY % CHUNK_SPTP == 0) -> no straddle
    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl((SP2, TP2))
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_SPTP, T_SPTP, seed=42)
        k_bc = _to_slab(k_nat, SP2, CHUNK_SPTP)  # block-cyclic physical layout the reader inverts

        # K: block-cyclic slab sharded on the SP axis (dim 2), replicated across TP.
        k_shard = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(SP2, TP2), dims=(2, None))
        k_local = ttnn.from_torch(k_bc, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=k_shard)
        # Gathered buffer: full T per device (replicated over both axes), zero-seeded.
        k_gathered = ttnn.from_torch(
            torch.zeros_like(k_nat),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # q/w: SP-shard seq (dim 2), then split those rows over TP (mesh_partition) -> each device owns Sq_sp/tp rows.
        qw_shard = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(SP2, TP2), dims=(2, None))
        q_dev = ttnn.from_torch(q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=qw_shard)
        w_dev = ttnn.from_torch(w_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=qw_shard)
        q_dev = ttnn.mesh_partition(q_dev, dim=2, cluster_axis=TP2_AXIS)
        w_dev = ttnn.mesh_partition(w_dev, dim=2, cluster_axis=TP2_AXIS)

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP2_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=2,
            ag_sub_device_id=subdevice_id,
            chunk_start_idx=chunk_start,
            seq_subshard_axis=TP2_AXIS,  # the SP×TP feature under test
            block_cyclic_sp_axis=SP2_AXIS,
            block_cyclic_chunk_local=chunk_local,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        # Compose device shards back to global chunk order (row-major: SP outer, TP inner).
        shards = [ttnn.to_torch(s) for s in ttnn.get_device_tensors(out.cpu())]
        out_t = torch.cat(shards, dim=2)

        ref = _per_sp_tp_ref(q_g, k_nat, w_g, SP2, TP2, chunk_start, chunk_local)
        assert_indexer_match(out_t, ref, CHUNK_SPTP, T_SPTP, check_neg=True)
        logger.info(f"SP×TP fused (heads={heads}): sp2 ring × tp2 seq sub-shard on FABRIC_2D matched reference")
    finally:
        _close_ccl(mesh)
