"""Launcher host-side del workflow DDP containerizzato News-M3F."""

import argparse
import json
import shlex
import shutil
import socket
import subprocess
from pathlib import Path


DEFAULT_CONFIG_PATH = Path("config/distributed.json")


def parse_args():
    """Legge la configurazione e gli eventuali override runtime."""

    parser = argparse.ArgumentParser(
        description=(
            "Avvia nel container Docker il workflow distribuito "
            "News-M3F sul nodo corrente."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Configurazione del cluster distribuito.",
    )

    parser.add_argument(
        "--master-addr",
        default=None,
        help=(
            "Override dell'indirizzo del rank 0. "
            "Se omesso viene usato il valore del file JSON."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override del numero massimo di epoche.",
    )

    parser.add_argument(
        "--output-root",
        default=None,
        help="Override della directory degli output nel container.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra il comando Docker senza eseguirlo.",
    )

    return parser.parse_args()


def load_config(config_path):
    """Carica e valida la configurazione essenziale del cluster."""

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
        "master_addr",
        "base_port",
        "local_batch_size",
        "gradient_accumulation_steps",
        "epochs",
        "docker_image",
        "output_root",
        "nodes",
    )

    for field in required_fields:
        if field not in config:
            raise ValueError(
                f"Campo '{field}' mancante in {config_path}."
            )

    return config


def resolve_node(hostname, nodes):
    """Determina rank e interfaccia di rete del nodo corrente."""

    normalized_nodes = {
        node_name.lower(): node_config
        for node_name, node_config in nodes.items()
    }

    normalized_hostname = hostname.lower()

    # Gestiamo anche l'eventualità che Linux restituisca
    # un hostname qualificato, ad esempio HP-NOTEBOOK.station.
    possible_names = (
        normalized_hostname,
        normalized_hostname.split(".")[0],
    )

    for name in possible_names:
        if name in normalized_nodes:
            node_config = normalized_nodes[name]

            if "rank" not in node_config:
                raise ValueError(
                    f"Rank mancante nella configurazione del nodo {hostname}."
                )

            if "interface" not in node_config:
                raise ValueError(
                    f"Interfaccia mancante nella configurazione "
                    f"del nodo {hostname}."
                )

            return node_config

    raise ValueError(
        f"Il nodo '{hostname}' non è presente nella configurazione. "
        f"Nodi configurati: {', '.join(nodes.keys())}"
    )


def main():
    args = parse_args()

    config = load_config(
        args.config
    )

    # Questo launcher deve essere eseguito sull'host WSL,
    # dove è disponibile il Docker Engine usato dal progetto.
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker non trovato. "
            "Eseguire il launcher dentro Ubuntu/WSL."
        )

    hostname = socket.gethostname()

    node_config = resolve_node(
        hostname,
        config["nodes"],
    )

    rank = node_config["rank"]
    network_interface = node_config["interface"]

    # Gli override permettono di cambiare parametri di deployment
    # senza modificare né il codice né necessariamente il JSON.
    master_addr = (
        args.master_addr
        if args.master_addr is not None
        else config["master_addr"]
    )

    epochs = (
        args.epochs
        if args.epochs is not None
        else config["epochs"]
    )

    output_root = (
        args.output_root
        if args.output_root is not None
        else config["output_root"]
    )

    # Ricaviamo automaticamente la radice del repository.
    # In questo modo non codifichiamo username o percorsi assoluti.
    repo_root = Path(__file__).resolve().parents[1]

    cache_dir = repo_root / "cache"
    outputs_dir = repo_root / "outputs"

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache non trovata: {cache_dir}"
        )

    outputs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    container_name = (
        f"lm32-workflow-rank{rank}"
    )

    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--name",
        container_name,
        "-e",
        f"GLOO_SOCKET_IFNAME={network_interface}",
        "-v",
        f"{cache_dir}:/app/cache:ro",
        "-v",
        f"{outputs_dir}:/app/outputs",
        config["docker_image"],
        "python",
        "-m",
        "src.run_fanout",
        "--rank",
        str(rank),
        "--world-size",
        str(config["world_size"]),
        "--master-addr",
        master_addr,
        "--base-port",
        str(config["base_port"]),
        "--epochs",
        str(epochs),
        "--local-batch-size",
        str(config["local_batch_size"]),
        "--grad-accum-steps",
        str(config["gradient_accumulation_steps"]),
        "--output-root",
        output_root,
    ]

    print("\n=== LAUNCHER WORKFLOW DISTRIBUITO ===")
    print(f"Hostname:          {hostname}")
    print(f"Rank:              {rank}")
    print(f"World size:        {config['world_size']}")
    print(f"Master address:    {master_addr}")
    print(f"Interfaccia Gloo:  {network_interface}")
    print(f"Porta base:        {config['base_port']}")
    print(f"Epoche:            {epochs}")
    print(f"Batch locale:      {config['local_batch_size']}")
    print(
        "Accumulo gradienti: "
        f"{config['gradient_accumulation_steps']}"
    )
    print(f"Immagine Docker:   {config['docker_image']}")
    print(f"Output root:       {output_root}")
    print(f"Repository:        {repo_root}")

    print("\nComando Docker:")
    print(
        shlex.join(command)
    )

    if args.dry_run:
        print(
            "\nDry run: container non avviato."
        )
        return

    subprocess.run(
        command,
        check=True,
    )


if __name__ == "__main__":
    main()