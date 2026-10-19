"""Launcher host-side del workflow DDP containerizzato News-M3F."""

import argparse
import json
import shlex
import shutil
import socket
import subprocess
from pathlib import Path


DEFAULT_CONFIG_PATH = Path("config/distributed.json")
CANDIDATES = ("A", "B", "C")


def parse_args():
    """Legge la configurazione e gli eventuali override runtime."""

    parser = argparse.ArgumentParser(
        description=(
            "Avvia il workflow distribuito News-M3F. "
            "Ogni candidato viene eseguito in un container indipendente."
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
        help="Override della directory radice degli output.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra i comandi Docker senza eseguirli.",
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

    possible_names = (
        normalized_hostname,
        normalized_hostname.split(".")[0],
    )

    for name in possible_names:
        if name in normalized_nodes:
            node_config = normalized_nodes[name]

            if "rank" not in node_config:
                raise ValueError(
                    f"Rank mancante per il nodo {hostname}."
                )

            if "interface" not in node_config:
                raise ValueError(
                    f"Interfaccia mancante per il nodo {hostname}."
                )

            return node_config

    raise ValueError(
        f"Il nodo '{hostname}' non è presente nella configurazione. "
        f"Nodi configurati: {', '.join(nodes.keys())}"
    )


def build_training_command(
    candidate,
    candidate_index,
    rank,
    network_interface,
    repo_root,
    config,
    master_addr,
    epochs,
    output_root,
):
    """Costruisce il docker run di un singolo ramo del Fan-out."""

    cache_dir = repo_root / "cache"
    outputs_dir = repo_root / "outputs"

    master_port = (
        config["base_port"] + candidate_index
    )

    candidate_output = (
        f"{output_root}/candidate_{candidate}"
    )

    container_name = (
        f"lm32-candidate-{candidate.lower()}-rank{rank}"
    )

    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--name",
        container_name,
        "-e",
        f"GLOO_SOCKET_IFNAME={network_interface}",
        "-e",
        "ATEN_CPU_CAPABILITY=default",
        "-e",
        "ONEDNN_MAX_CPU_ISA=AVX",
        "-e",
        "DNNL_MAX_CPU_ISA=AVX",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "MKL_NUM_THREADS=1",
        "-e",
        "OPENBLAS_NUM_THREADS=1",
        "-v",
        f"{cache_dir}:/app/cache:ro",
        "-v",
        f"{outputs_dir}:/app/outputs",
        config["docker_image"],
        "python",
        "-m",
        "src.train_ddp",
        "--candidate",
        candidate,
        "--epochs",
        str(epochs),
        "--local-batch-size",
        str(config["local_batch_size"]),
        "--grad-accum-steps",
        str(config["gradient_accumulation_steps"]),
        "--world-size",
        str(config["world_size"]),
        "--rank",
        str(rank),
        "--master-addr",
        master_addr,
        "--master-port",
        str(master_port),
        "--output-dir",
        candidate_output,
    ]


def build_fan_in_command(
    repo_root,
    config,
    output_root,
):
    """Costruisce il container dedicato al Fan-in finale."""

    outputs_dir = repo_root / "outputs"

    return [
        "docker",
        "run",
        "--rm",
        "--name",
        "lm32-fan-in",
        "-v",
        f"{outputs_dir}:/app/outputs",
        config["docker_image"],
        "python",
        "-m",
        "src.select_best",
        "--input-root",
        output_root,
    ]

def build_test_command(
    repo_root,
    config,
    output_root,
):
    """Costruisce il container dedicato alla valutazione finale sul test set."""

    cache_dir = repo_root / "cache"
    outputs_dir = repo_root / "outputs"

    return [
        "docker",
        "run",
        "--rm",
        "--name",
        "lm32-final-test",

        # Manteniamo la stessa configurazione CPU stabile
        # utilizzata durante il training sui nodi Ivy Bridge.
        "-e",
        "ATEN_CPU_CAPABILITY=default",
        "-e",
        "ONEDNN_MAX_CPU_ISA=AVX",
        "-e",
        "DNNL_MAX_CPU_ISA=AVX",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "MKL_NUM_THREADS=1",
        "-e",
        "OPENBLAS_NUM_THREADS=1",

        # Il test deve leggere sia il dataset preprocessato
        # sia il checkpoint selezionato dal Fan-in.
        "-v",
        f"{cache_dir}:/app/cache:ro",
        "-v",
        f"{outputs_dir}:/app/outputs",

        config["docker_image"],
        "python",
        "-m",
        "src.evaluate_test",
        "--input-root",
        output_root,
    ]

def main():
    args = parse_args()

    config = load_config(
        args.config
    )

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

    print("\n=== LAUNCHER WORKFLOW DISTRIBUITO ===")
    print(f"Hostname:           {hostname}")
    print(f"Rank:               {rank}")
    print(f"World size:         {config['world_size']}")
    print(f"Master address:     {master_addr}")
    print(f"Interfaccia Gloo:   {network_interface}")
    print(f"Porta base:         {config['base_port']}")
    print(f"Epoche:             {epochs}")
    print(f"Batch locale:       {config['local_batch_size']}")
    print(
        "Accumulo gradienti: "
        f"{config['gradient_accumulation_steps']}"
    )
    print(f"Immagine Docker:    {config['docker_image']}")
    print(f"Output root:        {output_root}")
    print(f"Repository:         {repo_root}")

    # FAN-OUT logico:
    # A, B e C sono rami indipendenti. Sul cluster disponibile
    # vengono schedulati in sequenza, ma ciascuno riceve un
    # container nuovo per isolare completamente le risorse native.
    for candidate_index, candidate in enumerate(CANDIDATES):

        master_port = (
            config["base_port"] + candidate_index
        )

        command = build_training_command(
            candidate=candidate,
            candidate_index=candidate_index,
            rank=rank,
            network_interface=network_interface,
            repo_root=repo_root,
            config=config,
            master_addr=master_addr,
            epochs=epochs,
            output_root=output_root,
        )

        print(
            f"\n=== FAN-OUT | CANDIDATO {candidate} "
            f"| RANK {rank} "
            f"| PORTA {master_port} ==="
        )

        print(
            shlex.join(command)
        )

        if not args.dry_run:
            subprocess.run(
                command,
                check=True,
            )

    # Il Fan-in non richiede DDP e viene eseguito una sola volta,
    # sul nodo rank 0, dopo il completamento dei tre rami.
    
    if rank == 0:
        fan_in_command = build_fan_in_command(
            repo_root=repo_root,
            config=config,
            output_root=output_root,
        )

        print(
            "\n=== FAN-OUT COMPLETATO: AVVIO FAN-IN ==="
        )

        print(
            shlex.join(fan_in_command)
        )

        if not args.dry_run:
            subprocess.run(
                fan_in_command,
                check=True,
            )

        # Il test set viene utilizzato soltanto dopo che il Fan-in
        # ha selezionato il modello migliore sulla validation.
        # Anche questa fase viene eseguita una sola volta sul rank 0.
        test_command = build_test_command(
            repo_root=repo_root,
            config=config,
            output_root=output_root,
        )

        print(
            "\n=== FAN-IN COMPLETATO: AVVIO TEST FINALE ==="
        )

        print(
            shlex.join(test_command)
        )

        if not args.dry_run:
            subprocess.run(
                test_command,
                check=True,
            )

    if args.dry_run:
        print(
            "\nDry run: nessun container avviato."
        )
    else:
        print(
            f"\n=== WORKFLOW COMPLETATO SUL RANK {rank} ==="
        )


if __name__ == "__main__":
    main()