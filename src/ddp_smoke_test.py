"""Smoke test di sharding e training distribuito con PyTorch DDP."""

from itertools import islice

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from src.data import load_preprocessed_data, make_collate_fn
from src.model import NewsClassifier


RANDOM_SEED = 42

WORLD_SIZE = 2
LOCAL_BATCH_SIZE = 64

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

SMOKE_TEST_BATCHES = 20

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"


def setup_distributed(rank, world_size):
    """Inizializza il gruppo distribuito usando il backend Gloo."""

    # Su Windows la build PyTorch CPU può non includere libuv.
    # Lo disabilitiamo esplicitamente nel TCPStore usato dal process group.
    init_method = (
        f"tcp://{MASTER_ADDR}:{MASTER_PORT}"
        "?use_libuv=0"
    )

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )


def cleanup_distributed():
    """Chiude ordinatamente il gruppo di processi distribuito."""

    dist.destroy_process_group()


def run_worker(rank, world_size):
    """Esegue lo smoke test relativo a un singolo worker DDP."""

    setup_distributed(rank, world_size)

    torch.manual_seed(RANDOM_SEED)

    dataset, config = load_preprocessed_data()

    # DistributedSampler implementa lo sharding logico:
    # i worker utilizzano record differenti dello stesso dataset.
    train_sampler = DistributedSampler(
        dataset["train"],
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=RANDOM_SEED,
        drop_last=True,
    )

    train_sampler.set_epoch(0)

    preview_indices = list(
        islice(iter(train_sampler), 5)
    )

    collate_fn = make_collate_fn(
        config["pad_token_id"]
    )

    train_loader = DataLoader(
        dataset["train"],
        batch_size=LOCAL_BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = NewsClassifier(
        vocabulary_size=config["vocabulary_size"],
        num_classes=config["num_classes"],
        pad_token_id=config["pad_token_id"],
    )

    # Ogni worker possiede una replica completa del modello.
    # DDP sincronizza automaticamente i gradienti durante backward().
    model = DDP(model)

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        f"\n[Rank {rank}] "
        f"world_size={world_size} | "
        f"shard={len(train_sampler):,} esempi | "
        f"primi indici={preview_indices}",
        flush=True,
    )

    total_loss = 0.0

    for batch_index, batch in enumerate(
        train_loader,
        start=1,
    ):
        optimizer.zero_grad()

        logits = model(
            batch["input_ids"],
            batch["attention_mask"],
        )

        loss = loss_function(
            logits,
            batch["labels"],
        )

        # In questo punto DDP esegue l'All-Reduce dei gradienti
        # calcolati indipendentemente dai due worker.
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch_index >= SMOKE_TEST_BATCHES:
            break

    average_loss = (
        total_loss / SMOKE_TEST_BATCHES
    )

    # Le due repliche devono rimanere identiche dopo gli
    # aggiornamenti effettuati con gradienti sincronizzati.
    parameter_checksum = (
        model.module.classifier.weight
        .detach()
        .sum()
        .item()
    )

    print(
        f"[Rank {rank}] "
        f"Loss media={average_loss:.4f} | "
        f"checksum={parameter_checksum:.6f}",
        flush=True,
    )

    dist.barrier()

    if rank == 0:
        print(
            "\n=== DDP SMOKE TEST COMPLETATO ===",
            flush=True,
        )
        print(
            "Sharding logico: OK\n"
            "Training multiprocesso: OK\n"
            "All-Reduce gradienti tramite DDP: OK",
            flush=True,
        )

    cleanup_distributed()


def main():
    # Su Windows viene utilizzato il metodo spawn:
    # vengono creati due processi Python indipendenti,
    # corrispondenti ai rank 0 e 1.
    mp.spawn(
        run_worker,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()