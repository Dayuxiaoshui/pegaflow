#!/usr/bin/env python3
"""Probe the native P/D RDMA binding without starting vLLM."""

from __future__ import annotations

import argparse
import json

from pegaflow.pegaflow import PdRdmaEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--numa-node", type=int)
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--device", choices=("cuda", "host"), default="cuda")
    args = parser.parse_args()

    engine = PdRdmaEngine(
        cuda_device=args.cuda_device,
        numa_node=args.numa_node,
        domains=args.domain or None,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "main_address": engine.main_address(),
                "num_domains": engine.num_domains(),
                "num_groups": engine.num_groups(),
                "aggregated_link_speed": engine.aggregated_link_speed(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
