"""Orchestrazione Fan-out/Fan-in del workflow distribuito News-M3F."""

import argparse
import subprocess
import sys
from pathlib import Path


CANDIDATES = ("A", "B", "C")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Esegue i candidati A/B/C in DDP e, sul rank 0, "
            "applica il Fan-in finale."
        )
    )

    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        choices=(0, 1),
        help="Rank DDP del nodo corrente.",
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--master-addr",
        required=True,
        help="Indirizzo o hostname del nodo rank 0.",
    )

    parser.add_argument(
        "--base-port",
        type=int,
        default=29500,
        help="Porta del candidato A; B e C usano le successive.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/distributed"),
        help="Directory radice degli output dei candidati.",
    )

    return parser.parse_args()


def run_candidate(candidate, candidate_index, args):
    """Avvia un ramo del Fan-out tramite train_ddp."""

    # Ogni candidato usa una porta diversa.
    # In questo modo un rank può attendere l'altro sul ramo successivo
    # senza interferire con il process group precedente.
    master_port = args.base_port + candidate_index

    output_dir = (
        args.output_root
        / f"candidate_{candidate}"
    )

    command = [
        sys.executable,
        "-m",
        "src.train_ddp",
        "--candidate",
        candidate,
        "--epochs",
        str(args.epochs),
        "--local-batch-size",
        str(args.local_batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--world-size",
        str(args.world_size),
        "--rank",
        str(args.rank),
        "--master-addr",
        args.master_addr,
        "--master-port",
        str(master_port),
        "--output-dir",
        str(output_dir),
    ]

    print(
        f"\n=== FAN-OUT | CANDIDATO {candidate} "
        f"| RANK {args.rank} "
        f"| PORTA {master_port} ===",
        flush=True,
    )

    # check=True interrompe l'intero workflow se un ramo fallisce:
    # non avrebbe senso eseguire il Fan-in con risultati incompleti.
    subprocess.run(
        command,
        check=True,
    )


def run_fan_in(args):
    """Esegue il Fan-in finale esclusivamente sul rank 0."""

    if args.rank != 0:
        return

    # select_best attualmente opera sulla directory
    # outputs/distributed dei run finali.
    if args.output_root != Path("outputs/distributed"):
        print(
            "\nFan-in automatico non eseguito: "
            "output-root diverso da outputs/distributed.",
            flush=True,
        )
        return

    print(
        "\n=== FAN-OUT COMPLETATO: AVVIO FAN-IN ===",
        flush=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.select_best",
        ],
        check=True,
    )


def main():
    args = parse_args()

    print("\n=== WORKFLOW DISTRIBUITO NEWS-M3F ===")
    print(f"Rank:              {args.rank}")
    print(f"World size:        {args.world_size}")
    print(f"Master:            {args.master_addr}")
    print(f"Porta base:        {args.base_port}")
    print(f"Epoche:            {args.epochs}")
    print(f"Batch locale:      {args.local_batch_size}")
    print(f"Accumulo gradienti:{args.grad_accum_steps}")
    print(f"Output root:       {args.output_root}")

    # FAN-OUT logico:
    # i tre candidati costituiscono rami indipendenti del workflow.
    # Poiché ciascun ramo utilizza entrambi i nodi disponibili,
    # vengono schedulati in sequenza anziché simultaneamente.
    for candidate_index, candidate in enumerate(CANDIDATES):
        run_candidate(
            candidate,
            candidate_index,
            args,
        )

    run_fan_in(args)

    print(
        f"\n=== WORKFLOW COMPLETATO SUL RANK {args.rank} ===",
        flush=True,
    )


if __name__ == "__main__":
    main()