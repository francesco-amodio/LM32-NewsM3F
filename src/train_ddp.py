"""Training distribuito del classificatore News-M3F con PyTorch DDP."""

import argparse
import json
import platform
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from src.data import load_preprocessed_data, make_collate_fn
from src.model import EMBEDDING_DIM, NewsClassifier
from src.train import (
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MIN_EPOCHS_BEFORE_STOPPING,
    MIN_IMPROVEMENT,
    PREDICTION_THRESHOLD,
    RANDOM_SEED,
    VALIDATION_BATCH_SIZE,
    WEIGHT_DECAY,
    build_loss_function,
    evaluate_model,
    set_random_seed,
)


DEFAULT_WORLD_SIZE = 2
DEFAULT_LOCAL_BATCH_SIZE = 64
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_MAX_EPOCHS = 8

DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = 29500

DEFAULT_OUTPUT_DIR = Path("outputs/ddp")


def parse_args():
    """Legge la configurazione del training distribuito."""

    parser = argparse.ArgumentParser(
        description="Training distribuito News-M3F con PyTorch DDP."
    )

    parser.add_argument(
        "--candidate",
        choices=["A", "B", "C"],
        default="A",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
    )

    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=DEFAULT_LOCAL_BATCH_SIZE,
        help="Batch elaborato da ciascun worker.",
    )

    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=DEFAULT_GRAD_ACCUM_STEPS,
        help=(
            "Numero di micro-batch da accumulare prima della "
            "sincronizzazione DDP e dell'aggiornamento dei pesi."
        ),
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=DEFAULT_WORLD_SIZE,
        help="Numero totale di worker DDP.",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help=(
            "Rank del worker. Se omesso, vengono creati localmente "
            "world-size processi tramite multiprocessing."
        ),
    )

    parser.add_argument(
        "--master-addr",
        default=DEFAULT_MASTER_ADDR,
        help="Indirizzo del worker rank 0.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=DEFAULT_MASTER_PORT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps deve essere almeno 1.")

    return args


def setup_distributed(rank, world_size, master_addr, master_port):
    """Inizializza il gruppo di processi DDP."""

    init_method = (
        f"tcp://{master_addr}:{master_port}"
    )

    # La build Windows di PyTorch usata nel progetto non dispone
    # del supporto libuv richiesto dal TCPStore predefinito.
    if platform.system() == "Windows":
        init_method += "?use_libuv=0"

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )


def cleanup_distributed():
    """Chiude il process group."""

    dist.destroy_process_group()


def train_one_epoch_ddp(
    model,
    loader,
    optimizer,
    loss_function,
    epoch_number,
    rank,
    grad_accum_steps,
):
    """Esegue un'epoca DDP con accumulo e loss globale dei worker."""

    model.train()

    local_loss_sum = 0.0
    local_examples = 0
    total_batches = len(loader)

    # I gradienti vengono azzerati all'inizio di ogni gruppo di
    # micro-batch. I micro-batch intermedi accumulano i gradienti
    # localmente senza eseguire All-Reduce.
    optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader, start=1):
        group_start = (
            ((batch_index - 1) // grad_accum_steps)
            * grad_accum_steps
            + 1
        )
        group_end = min(
            group_start + grad_accum_steps - 1,
            total_batches,
        )

        # L'ultimo gruppo può contenere meno micro-batch del previsto.
        # Calcoliamo quanti esempi contiene realmente, così la loss
        # accumulata resta correttamente pesata anche nel gruppo finale.
        group_size = group_end - group_start + 1
        nominal_batch_size = loader.batch_size
        last_batch_size = (
            len(loader.sampler)
            - (total_batches - 1) * nominal_batch_size
        )

        if group_end == total_batches:
            group_examples = (
                (group_size - 1) * nominal_batch_size
                + last_batch_size
            )
        else:
            group_examples = group_size * nominal_batch_size

        batch_size = batch["labels"].shape[0]
        should_sync = batch_index == group_end

        # DDP normalmente sincronizza i gradienti a ogni backward().
        # no_sync() evita la Collective Communication sui micro-batch
        # intermedi; l'ultimo backward del gruppo esegue l'All-Reduce.
        sync_context = (
            nullcontext()
            if should_sync
            else model.no_sync()
        )

        with sync_context:
            logits = model(
                batch["input_ids"],
                batch["attention_mask"],
            )

            loss = loss_function(
                logits,
                batch["labels"],
            )

            # La loss di ciascun micro-batch viene pesata in base al
            # numero di esempi che contiene. In questo modo i gradienti
            # accumulati equivalgono alla media dell'intero gruppo.
            scaled_loss = (
                loss
                * batch_size
                / group_examples
            )
            scaled_loss.backward()

        # L'optimizer aggiorna i pesi soltanto dopo avere accumulato
        # tutti i micro-batch del gruppo e sincronizzato i gradienti.
        if should_sync:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Per le statistiche conserviamo la loss originale, non quella
        # scalata per il gradient accumulation.
        local_loss_sum += loss.item() * batch_size
        local_examples += batch_size

        if rank == 0 and (
            batch_index % 100 == 0
            or batch_index == total_batches
        ):
            print(
                f"Epoca {epoch_number} | "
                f"Micro-batch locale "
                f"{batch_index}/{total_batches}",
                flush=True,
            )

    # Ogni worker possiede soltanto la loss del proprio shard.
    # Un ulteriore All-Reduce somma le statistiche locali per
    # ottenere la loss media dell'intero training distribuito.
    statistics = torch.tensor(
        [local_loss_sum, local_examples],
        dtype=torch.float64,
    )

    dist.all_reduce(
        statistics,
        op=dist.ReduceOp.SUM,
    )

    global_loss_sum = statistics[0].item()
    global_examples = int(statistics[1].item())

    return (
        global_loss_sum / global_examples,
        global_examples,
    )


def save_checkpoint(
    model,
    optimizer,
    candidate,
    epoch,
    metrics,
    output_dir,
):
    """Salva il miglior modello. Viene chiamato soltanto dal rank 0."""

    checkpoint_dir = (
        output_dir / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / f"candidate_{candidate}_best.pt"
    )

    torch.save(
        {
            "candidate": candidate,
            "epoch": epoch,

            # Salviamo il modello reale contenuto nel wrapper DDP,
            # evitando il prefisso "module." nelle chiavi del checkpoint.
            "model_state_dict": model.module.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )

    return checkpoint_path


def save_metadata(candidate, metadata, output_dir):
    """Salva i metadata prodotti dal rank 0."""

    metadata_dir = (
        output_dir / "metadata"
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        metadata_dir
        / f"candidate_{candidate}.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return metadata_path


def run_worker(rank, args):
    """Esegue un singolo worker del training DDP."""

    setup_distributed(
        rank=rank,
        world_size=args.world_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
    )

    set_random_seed(RANDOM_SEED)

    dataset, config = load_preprocessed_data()

    # DistributedSampler divide logicamente il training set
    # tra i worker senza creare copie fisiche separate del dataset.
    train_sampler = DistributedSampler(
        dataset["train"],
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True,
        seed=RANDOM_SEED,

        # Con 119.529 record e due worker eliminiamo un solo
        # record per ottenere shard esattamente della stessa dimensione.
        drop_last=True,
    )

    preview_indices = list(
        islice(iter(train_sampler), 5)
    )

    collate_fn = make_collate_fn(
        config["pad_token_id"]
    )

    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.local_batch_size,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # La validation viene eseguita soltanto dal rank 0.
    # Gli altri worker attendono il risultato prima
    # dell'eventuale epoca successiva.
    validation_loader = None

    if rank == 0:
        validation_loader = DataLoader(
            dataset["validation"],
            batch_size=VALIDATION_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    model = NewsClassifier(
        vocabulary_size=config["vocabulary_size"],
        num_classes=config["num_classes"],
        pad_token_id=config["pad_token_id"],
    )

    # Ogni processo mantiene una replica completa del modello.
    # DDP sincronizza automaticamente i gradienti tra le repliche.
    model = DDP(
        model,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )

    loss_function, pos_weight = build_loss_function(
        args.candidate,
        dataset["train"],
        config["num_classes"],
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        f"[Rank {rank}] "
        f"shard={len(train_sampler):,} | "
        f"primi indici={preview_indices}",
        flush=True,
    )

    metadata = None

    if rank == 0:
        total_parameters = sum(
            parameter.numel()
            for parameter in model.module.parameters()
        )

        metadata = {
            "candidate": args.candidate,
            "execution_mode": "ddp",
            "backend": "gloo",
            "world_size": args.world_size,
            "local_batch_size": args.local_batch_size,
            "global_batch_size": (
                args.local_batch_size
                * args.world_size
            ),
            "gradient_accumulation_steps": args.grad_accum_steps,
            "effective_global_batch_size": (
                args.local_batch_size
                * args.world_size
                * args.grad_accum_steps
            ),
            "random_seed": RANDOM_SEED,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "max_epochs": args.epochs,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "min_improvement": MIN_IMPROVEMENT,
            "embedding_dim": EMBEDDING_DIM,
            "vocabulary_size": config["vocabulary_size"],
            "num_classes": config["num_classes"],
            "model_parameters": total_parameters,
            "training_examples_original": len(
                dataset["train"]
            ),
            "training_examples_per_rank": len(
                train_sampler
            ),
            "training_examples_used": (
                len(train_sampler)
                * args.world_size
            ),
            "history": [],
        }

        if pos_weight is None:
            metadata["pos_weight"] = None
        else:
            metadata["pos_weight"] = {
                "min": float(pos_weight.min().item()),
                "max": float(pos_weight.max().item()),
                "mean": float(pos_weight.mean().item()),
            }

        print("\n=== CONFIGURAZIONE DDP ===")
        print(f"Candidato:          {args.candidate}")
        print(f"World size:         {args.world_size}")
        print(
            f"Batch locale:       "
            f"{args.local_batch_size}"
        )
        print(
            f"Batch globale:      "
            f"{args.local_batch_size * args.world_size}"
        )
        print(
            f"Accumulo gradienti: "
            f"{args.grad_accum_steps}"
        )
        print(
            f"Batch globale eff.: "
            f"{args.local_batch_size * args.world_size * args.grad_accum_steps}"
        )
        print(
            f"Esempi per worker:  "
            f"{len(train_sampler):,}"
        )
        print(
            f"Esempi utilizzati:  "
            f"{len(train_sampler) * args.world_size:,}"
        )

    best_micro_f1 = -1.0
    best_epoch = 0
    best_checkpoint = None
    epochs_without_improvement = 0

    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        # Tutti i worker devono utilizzare lo stesso numero di epoca,
        # affinché lo shuffle distribuito sia coerente e riproducibile.
        train_sampler.set_epoch(epoch - 1)

        epoch_start = time.perf_counter()

        if rank == 0:
            print(
                f"\n=== EPOCA {epoch}/{args.epochs} ===",
                flush=True,
            )

        train_loss, training_examples = train_one_epoch_ddp(
            model,
            train_loader,
            optimizer,
            loss_function,
            epoch,
            rank,
            args.grad_accum_steps,
        )

        stop_training = False

        if rank == 0:
            # Valutiamo il modello sottostante e non il wrapper DDP:
            # la validation non richiede comunicazione tra i worker.
            metrics = evaluate_model(
                model.module,
                validation_loader,
                loss_function,
            )

            epoch_seconds = (
                time.perf_counter() - epoch_start
            )

            epoch_result = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": metrics["validation_loss"],
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "predicted_positives": metrics[
                    "predicted_positives"
                ],
                "actual_positives": metrics[
                    "actual_positives"
                ],
                "mean_probability": metrics[
                    "mean_probability"
                ],
                "max_probability": metrics[
                    "max_probability"
                ],
                "duration_seconds": epoch_seconds,
                "training_examples": training_examples,
            }

            metadata["history"].append(
                epoch_result
            )

            print(f"Training loss:       {train_loss:.4f}")
            print(
                f"Validation loss:     "
                f"{metrics['validation_loss']:.4f}"
            )
            print(
                f"Micro-F1:            "
                f"{metrics['micro_f1']:.4f}"
            )
            print(
                f"Macro-F1:            "
                f"{metrics['macro_f1']:.4f}"
            )
            print(
                f"Precision:           "
                f"{metrics['precision']:.4f}"
            )
            print(
                f"Recall:              "
                f"{metrics['recall']:.4f}"
            )
            print(
                f"Durata epoca:        "
                f"{epoch_seconds:.1f} s"
            )

            improvement = (
                metrics["micro_f1"]
                - best_micro_f1
            )

            if improvement > MIN_IMPROVEMENT:
                best_micro_f1 = metrics["micro_f1"]
                best_epoch = epoch
                epochs_without_improvement = 0

                best_checkpoint = save_checkpoint(
                    model,
                    optimizer,
                    args.candidate,
                    epoch,
                    metrics,
                    args.output_dir,
                )

                print(
                    "Nuovo miglior modello: "
                    f"{best_checkpoint}"
                )

            else:
                epochs_without_improvement += 1

            if (
                epoch >= MIN_EPOCHS_BEFORE_STOPPING
                and epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                stop_training = True

        # Il rank 0 comunica agli altri worker se devono
        # continuare o interrompere il ciclo di training.
        stop_tensor = torch.tensor(
            [1 if stop_training else 0],
            dtype=torch.int32,
        )

        dist.broadcast(
            stop_tensor,
            src=0,
        )

        if stop_tensor.item() == 1:
            if rank == 0:
                print(
                    "\nEarly stopping.",
                    flush=True,
                )
            break

    if rank == 0:
        total_seconds = (
            time.perf_counter() - training_start
        )

        metadata["epochs_completed"] = len(
            metadata["history"]
        )
        metadata["best_epoch"] = best_epoch
        metadata["best_micro_f1"] = best_micro_f1
        metadata["best_checkpoint"] = (
            str(best_checkpoint)
            if best_checkpoint is not None
            else None
        )
        metadata["total_training_seconds"] = total_seconds

        metadata_path = save_metadata(
            args.candidate,
            metadata,
            args.output_dir,
        )

        print("\n=== TRAINING DDP COMPLETATO ===")
        print(f"Migliore epoca:    {best_epoch}")
        print(f"Migliore Micro-F1: {best_micro_f1:.4f}")
        print(f"Durata totale:     {total_seconds:.1f} s")
        print(f"Metadata:          {metadata_path}")

    # Evita che un worker distrugga il process group
    # mentre un altro sta ancora terminando.
    dist.barrier()

    cleanup_distributed()


def main():
    args = parse_args()

    if args.rank is None:
        # Modalità locale: crea tutti i worker sullo stesso computer.
        mp.spawn(
            run_worker,
            args=(args,),
            nprocs=args.world_size,
            join=True,
        )

    else:
        # Modalità multi-nodo: il processo è già associato
        # esplicitamente a un rank.
        run_worker(
            args.rank,
            args,
        )


if __name__ == "__main__":
    main()