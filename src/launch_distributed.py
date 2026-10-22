"""Launcher host-side del workflow DDP containerizzato News-M3F."""

import argparse
import json
import re
import shlex
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path


DEFAULT_CONFIG_PATH = Path("config/distributed.json")
CANDIDATES = ("A", "B", "C")
RUN_SYNC_TIMEOUT_SECONDS = 120


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
        "--run-description",
        default=None,
        help=(
            "Descrizione opzionale del run. Il rank 0 la aggiunge "
            "al timestamp usato come nome della directory di output."
        ),
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


def sanitize_run_description(description):
    """Rende sicura la descrizione per il nome di una directory."""

    if description is None:
        return None

    description = re.sub(r"\s+", " ", description).strip()

    if not description:
        return None

    # Evita caratteri non validi nei nomi di directory Windows.
    description = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]',
        "_",
        description,
    )
    description = description.rstrip(" .")

    return description or None


def build_run_name(run_description):
    """Genera il nome del run usando esclusivamente l'orologio del rank 0."""

    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    run_description = sanitize_run_description(run_description)

    if run_description:
        return (
            f"{timestamp} - {run_description}",
            run_description,
        )

    return timestamp, None


def publish_run_identity(
    run_name,
    run_description,
    sync_port,
    world_size,
):
    """Invia agli altri rank il nome del run deciso dal rank 0."""

    payload = (
        json.dumps(
            {
                "run_name": run_name,
                "run_description": run_description,
            },
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as server:
        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        server.bind(("", sync_port))
        server.listen(max(1, world_size - 1))

        print(
            "Rank 0 in attesa dei peer sulla porta "
            f"{sync_port} per condividere il nome del run...",
            flush=True,
        )

        for _ in range(world_size - 1):
            connection, _ = server.accept()

            with connection:
                connection.sendall(payload)


def receive_run_identity(
    master_addr,
    sync_port,
    timeout_seconds=RUN_SYNC_TIMEOUT_SECONDS,
):
    """Riceve dal rank 0 il nome condiviso del run."""

    deadline = time.monotonic() + timeout_seconds
    last_error = None

    while time.monotonic() < deadline:
        try:
            with socket.create_connection(
                (master_addr, sync_port),
                timeout=5,
            ) as client:
                with client.makefile(
                    "r",
                    encoding="utf-8",
                ) as stream:
                    line = stream.readline()

                if not line:
                    raise RuntimeError(
                        "Il rank 0 ha chiuso la connessione senza "
                        "inviare il nome del run."
                    )

                return json.loads(line)

        except (
            ConnectionRefusedError,
            OSError,
            json.JSONDecodeError,
        ) as error:
            last_error = error
            time.sleep(1)

    raise TimeoutError(
        "Impossibile ricevere dal rank 0 il nome del run su "
        f"{master_addr}:{sync_port}."
    ) from last_error


def resolve_run_identity(
    rank,
    world_size,
    master_addr,
    base_port,
    requested_description,
    synchronize=True,
):
    """Fa generare il nome al rank 0 e lo distribuisce agli altri rank."""

    requested_description = sanitize_run_description(
        requested_description
    )

    # In dry-run non viene eseguito alcun workflow reale: evitiamo quindi
    # di obbligare alla presenza simultanea di entrambi i nodi.
    if not synchronize:
        return build_run_name(requested_description)

    # Le porte base, base+1 e base+2 sono usate dai candidati A, B e C.
    # base-1 viene riservata al breve handshake iniziale del launcher.
    sync_port = base_port - 1

    if sync_port <= 0:
        raise ValueError(
            "base_port deve essere maggiore di 1."
        )

    if rank == 0:
        run_name, run_description = build_run_name(
            requested_description
        )

        if world_size > 1:
            publish_run_identity(
                run_name=run_name,
                run_description=run_description,
                sync_port=sync_port,
                world_size=world_size,
            )

        return run_name, run_description

    payload = receive_run_identity(
        master_addr=master_addr,
        sync_port=sync_port,
    )

    run_name = payload["run_name"]
    run_description = payload.get("run_description")

    # Il rank 0 è la fonte autorevole del nome. Se anche sul rank 1
    # è stata specificata una descrizione, verifichiamo che coincida.
    if (
        requested_description is not None
        and requested_description != run_description
    ):
        raise ValueError(
            "La descrizione del run specificata su questo nodo "
            "non coincide con quella determinata dal rank 0."
        )

    return run_name, run_description


def build_training_command(
    candidate,
    candidate_index,
    rank,
    network_interface,
    repo_root,
    config,
    master_addr,
    epochs,
    run_output_root,
):
    """Costruisce il docker run di un singolo ramo del Fan-out."""

    cache_dir = repo_root / "cache"
    outputs_dir = repo_root / "outputs"

    master_port = (
        config["base_port"] + candidate_index
    )

    candidate_output = (
        f"{run_output_root}/candidate_{candidate}"
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
    run_output_root,
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
        run_output_root,
    ]


def build_test_command(
    repo_root,
    config,
    run_output_root,
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
        run_output_root,
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

    run_name, run_description = resolve_run_identity(
        rank=rank,
        world_size=config["world_size"],
        master_addr=master_addr,
        base_port=config["base_port"],
        requested_description=args.run_description,
        synchronize=not args.dry_run,
    )

    run_output_root = f"outputs/{run_name}"

    # La directory viene creata su entrambi i nodi. Il nome è identico
    # perché, nelle esecuzioni reali, viene determinato dal solo rank 0.
    run_dir = outputs_dir / run_name
    run_dir.mkdir(
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
    print(f"Nome run:           {run_name}")
    print(
        "Descrizione run:    "
        f"{run_description or '(nessuna)'}"
    )
    print(f"Directory output:   {run_output_root}")
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
            run_output_root=run_output_root,
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
            run_output_root=run_output_root,
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
            run_output_root=run_output_root,
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
