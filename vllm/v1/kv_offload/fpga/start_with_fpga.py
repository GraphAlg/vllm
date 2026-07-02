#!/usr/bin/env python
"""
Start vLLM with FPGA KV cache offload (using the kv_offload abstraction).

Usage
-----
# Real hardware:
python start_with_fpga.py --model meta-llama/Llama-3.1-8B

# Mock mode (test without FPGA hardware):
python start_with_fpga.py --model meta-llama/Llama-3.1-8B --mock

# Direct server launch with JSON config:
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-8B \\
    --enable-prefix-caching \\
    --kv-transfer-config '{
        "kv_connector": "OffloadingConnector",
        "kv_connector_extra_config": {
            "spec_name": "FPGAOffloadingSpec",
            "xdma_prefix": "/dev/xdma0",
            "fpga_bytes_to_use": 34359738368,
            "eviction_policy": "lru"
        }
    }'
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Start vLLM with FPGA offload")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--xdma-prefix", default="/dev/xdma0")
    parser.add_argument("--fpga-size-gb", type=int, default=32)
    parser.add_argument("--mock", action="store_true",
                        help="Use mock XDMA (testing without hardware)")
    parser.add_argument("--eviction-policy", default="lru", choices=["lru", "arc"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args, remaining = parser.parse_known_args()

    # ── Step 1: Register the FPGA spec in OffloadingSpecFactory ────────
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory
    OffloadingSpecFactory.register_spec(
        "FPGAOffloadingSpec",
        "vllm.v1.kv_offload.fpga.spec",
        "FPGAOffloadingSpec",
    )

    # ── Step 2: Mock mode ──────────────────────────────────────────────
    if args.mock:
        os.environ["VLLM_FPGA_MOCK"] = "1"

    # ── Step 3: Build kv-transfer-config ────────────────────────────────
    fpga_bytes = args.fpga_size_gb * (1024**3)
    kv_transfer_config = json.dumps({
        "kv_connector": "OffloadingConnector",
        "kv_connector_extra_config": {
            "spec_name": "FPGAOffloadingSpec",
            "xdma_prefix": args.xdma_prefix,
            "fpga_bytes_to_use": fpga_bytes,
            "eviction_policy": args.eviction_policy,
        },
    })

    # ── Step 4: Launch vLLM server ─────────────────────────────────────
    launch_args = [
        "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--enable-prefix-caching",
        f"--gpu-memory-utilization={args.gpu_memory_utilization}",
        f"--kv-transfer-config", kv_transfer_config,
    ] + remaining

    # Set env vars for easy reference.
    os.environ["VLLM_FPGA_XDMA_PREFIX"] = args.xdma_prefix
    os.environ["VLLM_FPGA_DRAM_BYTES"] = str(fpga_bytes)

    sys.argv = launch_args
    from vllm.entrypoints.openai import api_server
    api_server.run_cmd()


if __name__ == "__main__":
    main()
