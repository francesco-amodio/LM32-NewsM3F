"""Avvio configurabile di un nodo del training distribuito DDP."""

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG_PATH = Path("config/distributed.json")


def parse_args():
    """Legge i parametri dell'esperimento da inoltrare al training DDP."""

    parser = argparse.ArgumentParser(
        description="Avvia il worker DDP configurato per questo nodo."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="File JSON con la configurazione del cluster.",
    )

    parser.add_argument(
        "--candidate",
        choices=["A", "B", "C"],
        default="A",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/ddp_multinode",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra il comando risolto senza avviare il training.",
    )

    return parser.parse_args()


def load_config(config_path):
    """Carica e valida i campi essenziali della configurazione distribuita."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configurazione non trovata: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    required_fields = (
        "backend",
        "world_size",
        "master_host",
        "master_port",
        "local_batch_size",
        "nodes",
    )

    for field in required_fields:
        if field not in config:
            raise ValueError(
                f"Campo '{field}' mancante in {config_path}."
            )

    return config


def resolve_rank(hostname, nodes):
    """Determina il rank associato al nome del computer."""

    # Confronto case-insensitive, utile soprattutto su Windows.
    normalized_nodes = {
        node_name.lower(): rank
        for node_name, rank in nodes.items()
    }

    normalized_hostname = hostname.lower()

    if normalized_hostname not in normalized_nodes:
        raise ValueError(
            f"Il nodo '{hostname}' non è presente nella configurazione. "
            f"Nodi configurati: {', '.join(nodes.keys())}"
        )

    return normalized_nodes[normalized_hostname]


def main():
    args = parse_args()

    config = load_config(
        args.config
    )

    hostname = socket.gethostname()

    rank = resolve_rank(
        hostname,
        config["nodes"],
    )

    command = [
        sys.executable,
        "-m",
        "src.train_ddp",
        "--candidate",
        args.candidate,
        "--epochs",
        str(args.epochs),
        "--world-size",
        str(config["world_size"]),
        "--local-batch-size",
        str(config["local_batch_size"]),
        "--rank",
        str(rank),
        "--master-addr",
        config["master_host"],
        "--master-port",
        str(config["master_port"]),
        "--output-dir",
        args.output_dir,
    ]

    print("\n=== CONFIGURAZIONE NODO ===")
    print(f"Hostname:       {hostname}")
    print(f"Rank:           {rank}")
    print(f"World size:     {config['world_size']}")
    print(f"Master:         {config['master_host']}")
    print(f"Master port:    {config['master_port']}")
    print(f"Backend:        {config['backend']}")
    print(
        f"Batch locale:   "
        f"{config['local_batch_size']}"
    )

    print("\nComando DDP:")
    print(" ".join(command))

    if args.dry_run:
        print("\nDry run: training non avviato.")
        return

    # Avvia train_ddp come processo figlio e restituisce
    # lo stesso codice di uscita in caso di errore.
    completed_process = subprocess.run(
        command,
        check=False,
    )

    if completed_process.returncode != 0:
        raise SystemExit(
            completed_process.returncode
        )


if __name__ == "__main__":
    main()