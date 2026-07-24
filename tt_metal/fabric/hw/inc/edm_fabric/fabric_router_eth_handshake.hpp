// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"

#if defined(COMPILE_FOR_AERISC)
// For WATCHER_RING_BUFFER_PUSH() used by the receiver-side local_value probe below. No-op unless the
// watcher is enabled (TT_METAL_WATCHER), so it costs nothing in production. Guarded to device eth builds.
#include "api/debug/ring_buffer.h"
#endif

namespace erisc {
namespace datamover {
namespace handshake {

// [RX-VALUE PROBE] Ring-buffer tag for the receiver handshake's first observed non-zero local_value.
// Pushed once per handshake attempt, low byte = the value seen. Absence of any 0x5E5EBBxx on a wedged
// receiver core => local_value never moved off 0 => the peer's remote-write never reached our L1
// (dead/diverted datapath). A 0x5E5EBBxx with xx != AA => the write lands but with a wrong value.
constexpr uint32_t FABRIC_DBG_HS_RX_VALUE_TAG = 0x5E5EBB00;

/*
 * Fabric-specific handshake functions with termination signal support.
 * These extend the base EDM handshake with the ability to exit early
 * when the host requests immediate termination, enabling graceful
 * recovery during fabric init/teardown.
 */

template <bool RISC_CPU_DATA_CACHE_ENABLED>
FORCE_INLINE void fabric_sender_side_handshake(
    uint32_t handshake_register_address,
    uint16_t my_mesh_id,
    uint8_t my_device_id,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    size_t HS_CONTEXT_SWITCH_TIMEOUT = A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH) {
    volatile tt_l1_ptr handshake_info_t* handshake_info =
        init_handshake_info(handshake_register_address, my_mesh_id, my_device_id);
    uint32_t local_val_addr = ((uint32_t)(&handshake_info->local_value)) / tt::tt_fabric::PACKET_WORD_SIZE_BYTES;
    uint32_t scratch_addr = ((uint32_t)(&handshake_info->scratch)) / tt::tt_fabric::PACKET_WORD_SIZE_BYTES;
    uint32_t count = 0;
    while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE
#ifndef ARCH_WORMHOLE
           && !tt::tt_fabric::got_immediate_termination_signal<RISC_CPU_DATA_CACHE_ENABLED>(termination_signal_ptr)
#endif
    ) {
        if (count == HS_CONTEXT_SWITCH_TIMEOUT) {
            count = 0;

#if (defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)) || !defined(ARCH_BLACKHOLE)
            run_routing();
#endif
        } else {
            count++;
            internal_::eth_send_packet(0, scratch_addr, local_val_addr, 1);
        }
        invalidate_l1_cache();
    }
}

template <bool RISC_CPU_DATA_CACHE_ENABLED>
FORCE_INLINE void fabric_receiver_side_handshake(
    uint32_t handshake_register_address,
    uint16_t my_mesh_id,
    uint8_t my_device_id,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    size_t HS_CONTEXT_SWITCH_TIMEOUT = A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH) {
    volatile tt_l1_ptr handshake_info_t* handshake_info =
        init_handshake_info(handshake_register_address, my_mesh_id, my_device_id);
    uint32_t local_val_addr = ((uint32_t)(&handshake_info->local_value)) / tt::tt_fabric::PACKET_WORD_SIZE_BYTES;
    uint32_t scratch_addr = ((uint32_t)(&handshake_info->scratch)) / tt::tt_fabric::PACKET_WORD_SIZE_BYTES;
    uint32_t count = 0;
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    // [RX-VALUE PROBE] Did the peer's remote-write ever land anything in our local_value? Capture the
    // FIRST non-zero value we observe (one-shot per attempt). Stuck at 0 forever => write never reached
    // this L1; non-zero-but-not-0xAA => it lands but wrong. See FABRIC_DBG_HS_RX_VALUE_TAG above.
    bool rx_value_probe_done = false;
#endif
    while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE
#ifndef ARCH_WORMHOLE
           && !tt::tt_fabric::got_immediate_termination_signal<RISC_CPU_DATA_CACHE_ENABLED>(termination_signal_ptr)
#endif
    ) {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
        if (!rx_value_probe_done && handshake_info->local_value != 0) {
            rx_value_probe_done = true;
            WATCHER_RING_BUFFER_PUSH(FABRIC_DBG_HS_RX_VALUE_TAG | (handshake_info->local_value & 0xFF));
        }
#endif
        if (count == HS_CONTEXT_SWITCH_TIMEOUT) {
            count = 0;

#if (defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)) || !defined(ARCH_BLACKHOLE)
            run_routing();
#endif
        } else {
            count++;
        }
        invalidate_l1_cache();
    }
#ifndef ARCH_WORMHOLE
    if (!tt::tt_fabric::got_immediate_termination_signal<RISC_CPU_DATA_CACHE_ENABLED>(termination_signal_ptr)) {
        internal_::eth_send_packet(0, scratch_addr, local_val_addr, 1);
    }
#else
    internal_::eth_send_packet(0, scratch_addr, local_val_addr, 1);
#endif
}

}  // namespace handshake
}  // namespace datamover
}  // namespace erisc
