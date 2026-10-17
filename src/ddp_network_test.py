"""Test minimale della comunicazione PyTorch Distributed tra due nodi."""

import argparse
import os
import platform
import socket
from datetime import timedelta

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--interface", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["GLOO_SOCKET_IFNAME"] = args.interface

    # Risolviamo esplicitamente l'hostname prima di passarlo a c10d.
    master_ip = socket.gethostbyname(args.master_addr)

    init_method = (
        f"tcp://{master_ip}:{args.master_port}"
    )

    if platform.system() == "Windows":
        init_method += "?use_libuv=0"

    print(
        f"[Rank {args.rank}] "
        f"hostname={socket.gethostname()} | "
        f"interface={args.interface} | "
        f"master={args.master_addr}->{master_ip}",
        flush=True,
    )

    print(
        f"[Rank {args.rank}] PRIMA di init_process_group",
        flush=True,
    )

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=args.rank,
        world_size=args.world_size,
        timeout=timedelta(seconds=30),
    )

    print(
        f"[Rank {args.rank}] DOPO init_process_group",
        flush=True,
    )

    print(
        f"[Rank {args.rank}] PRIMA della barrier",
        flush=True,
    )

    dist.barrier()

    print(
        f"[Rank {args.rank}] DOPO la barrier",
        flush=True,
    )

    # Rank 0 parte da 1, rank 1 parte da 2.
    # Dopo All-Reduce SUM entrambi devono ottenere 3.
    value = torch.tensor(
        [float(args.rank + 1)]
    )

    print(
        f"[Rank {args.rank}] "
        f"PRIMA All-Reduce: {value.item()}",
        flush=True,
    )

    dist.all_reduce(
        value,
        op=dist.ReduceOp.SUM,
    )

    print(
        f"[Rank {args.rank}] "
        f"DOPO All-Reduce: {value.item()}",
        flush=True,
    )

    dist.destroy_process_group()

    print(
        f"[Rank {args.rank}] TEST COMPLETATO",
        flush=True,
    )


if __name__ == "__main__":
    main()