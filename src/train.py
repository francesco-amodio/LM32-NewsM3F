"""Training e validazione del classificatore News-M3F."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data import load_preprocessed_data, make_collate_fn
from src.model import EMBEDDING_DIM, NewsClassifier


RANDOM_SEED = 42

# Nel training single-process il batch globale coincide con il batch locale.
# In DDP manterremo lo stesso batch globale: 64 esempi per ciascuno dei 2 worker.
DEFAULT_BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 128

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PREDICTION_THRESHOLD = 0.50

DEFAULT_MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 2
MIN_IMPROVEMENT = 0.001
MIN_EPOCHS_BEFORE_STOPPING = 4

OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METADATA_DIR = OUTPUT_DIR / "metadata"


def parse_args():
    """Legge i parametri principali dell'esperimento."""

    parser = argparse.ArgumentParser(
        description="Training del classificatore News-M3F."
    )

    parser.add_argument(
        "--candidate",
        choices=["A", "B", "C"],
        default="A",
        help=(
            "A: BCE non pesata; "
            "B: pos_weight=Nneg/Npos; "
            "C: pos_weight=sqrt(Nneg/Npos)."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="Numero massimo di epoche.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Dimensione del batch di training.",
    )

    return parser.parse_args()


def set_random_seed(seed):
    """Fissa i principali generatori casuali per la riproducibilità."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_positive_counts(train_dataset, num_classes):
    """Conta gli esempi positivi per ogni classe."""

    positive_counts = np.zeros(
        num_classes,
        dtype=np.int64,
    )

    for labels in train_dataset["labels"]:
        positive_counts += np.asarray(
            labels,
            dtype=np.int64,
        )

    return positive_counts


def build_loss_function(candidate, train_dataset, num_classes):
    """Costruisce la BCE corrispondente al candidato A, B o C."""

    if candidate == "A":
        return nn.BCEWithLogitsLoss(), None

    positive_counts = compute_positive_counts(
        train_dataset,
        num_classes,
    )

    total_examples = len(train_dataset)
    negative_counts = total_examples - positive_counts

    safe_positive_counts = np.maximum(
        positive_counts,
        1,
    )

    full_weights = (
        negative_counts / safe_positive_counts
    ).astype(np.float32)

    if candidate == "B":
        weights = full_weights

    elif candidate == "C":
        # La radice quadrata attenua i pesi estremi del candidato B.
        weights = np.sqrt(full_weights).astype(np.float32)

    else:
        raise ValueError(
            f"Candidato non riconosciuto: {candidate}"
        )

    pos_weight = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    return (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        pos_weight,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_function,
    epoch_number,
):
    """Esegue una singola epoca completa di training."""

    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch_index, batch in enumerate(loader, start=1):
        optimizer.zero_grad()

        logits = model(
            batch["input_ids"],
            batch["attention_mask"],
        )

        loss = loss_function(
            logits,
            batch["labels"],
        )

        loss.backward()
        optimizer.step()

        batch_size = batch["labels"].shape[0]

        total_loss += loss.item() * batch_size
        total_examples += batch_size

        # Stampiamo un aggiornamento periodico senza produrre
        # una riga di log per ogni singolo batch.
        if batch_index % 100 == 0:
            current_average = total_loss / total_examples

            print(
                f"Epoca {epoch_number} | "
                f"Batch {batch_index}/{len(loader)} | "
                f"Loss media: {current_average:.4f}"
            )

    return total_loss / total_examples


def evaluate_model(
    model,
    loader,
    loss_function,
    threshold=PREDICTION_THRESHOLD,
):
    """Valuta il modello sull'intero validation set."""

    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_targets = []
    all_predictions = []

    probability_sum = 0.0
    probability_count = 0
    maximum_probability = 0.0

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"],
                batch["attention_mask"],
            )

            loss = loss_function(
                logits,
                batch["labels"],
            )

            batch_size = batch["labels"].shape[0]

            total_loss += loss.item() * batch_size
            total_examples += batch_size

            probabilities = torch.sigmoid(logits)

            # Questi valori diagnostici ci aiutano a capire se un modello
            # è vicino alla soglia anche quando non produce ancora positivi.
            probability_sum += probabilities.sum().item()
            probability_count += probabilities.numel()
            maximum_probability = max(
                maximum_probability,
                probabilities.max().item(),
            )

            predictions = (
                probabilities >= threshold
            ).to(torch.int32)

            all_predictions.append(
                predictions.cpu()
            )

            all_targets.append(
                batch["labels"]
                .to(torch.int32)
                .cpu()
            )

    targets = torch.cat(all_targets).numpy()
    predictions = torch.cat(all_predictions).numpy()

    return {
        "validation_loss": total_loss / total_examples,
        "micro_f1": f1_score(
            targets,
            predictions,
            average="micro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "precision": precision_score(
            targets,
            predictions,
            average="micro",
            zero_division=0,
        ),
        "recall": recall_score(
            targets,
            predictions,
            average="micro",
            zero_division=0,
        ),
        "predicted_positives": int(predictions.sum()),
        "actual_positives": int(targets.sum()),
        "mean_probability": probability_sum / probability_count,
        "max_probability": maximum_probability,
    }


def save_checkpoint(
    model,
    optimizer,
    candidate,
    epoch,
    metrics,
):
    """Salva lo stato del miglior modello osservato."""

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"candidate_{candidate}_best.pt"
    )

    torch.save(
        {
            "candidate": candidate,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )

    return checkpoint_path


def save_metadata(candidate, metadata):
    """Salva configurazione e risultati dell'esperimento in JSON."""

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        METADATA_DIR
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


def main():
    args = parse_args()

    set_random_seed(RANDOM_SEED)

    dataset, config = load_preprocessed_data()

    collate_fn = make_collate_fn(
        config["pad_token_id"]
    )

    # Un generatore esplicitamente inizializzato rende riproducibile
    # anche l'ordine casuale degli esempi nel DataLoader.
    train_generator = torch.Generator()
    train_generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        generator=train_generator,
    )

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

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    metadata = {
        "candidate": args.candidate,
        "random_seed": RANDOM_SEED,
        "batch_size": args.batch_size,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
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

    print("\n=== CONFIGURAZIONE TRAINING ===")
    print(f"Candidato:          {args.candidate}")
    print(f"Epoche massime:     {args.epochs}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Training examples:  {len(dataset['train']):,}")
    print(f"Validation:         {len(dataset['validation']):,}")
    print(f"Parametri modello:  {total_parameters:,}")

    if pos_weight is None:
        print("pos_weight:         nessuno")
    else:
        print(
            f"pos_weight:         "
            f"{pos_weight.min().item():.2f} - "
            f"{pos_weight.max().item():.2f}"
        )

    best_micro_f1 = -1.0
    best_epoch = 0
    best_checkpoint = None
    epochs_without_improvement = 0

    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        print(
            f"\n=== EPOCA {epoch}/{args.epochs} ==="
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            epoch,
        )

        metrics = evaluate_model(
            model,
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
            "predicted_positives": metrics["predicted_positives"],
            "actual_positives": metrics["actual_positives"],
            "mean_probability": metrics["mean_probability"],
            "max_probability": metrics["max_probability"],
            "duration_seconds": epoch_seconds,
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
            f"Predizioni positive: "
            f"{metrics['predicted_positives']:,}"
        )
        print(
            f"Probabilità media:   "
            f"{metrics['mean_probability']:.4f}"
        )
        print(
            f"Probabilità massima: "
            f"{metrics['max_probability']:.4f}"
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
            )

            print(
                "Nuovo miglior modello: "
                f"checkpoint salvato in {best_checkpoint}"
            )

        else:
            epochs_without_improvement += 1

        # Evitiamo che una fase iniziale con Micro-F1 nullo
        # interrompa troppo presto l'addestramento.
        if (
            epoch >= MIN_EPOCHS_BEFORE_STOPPING
            and epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping: nessun miglioramento "
                "sufficiente della Micro-F1."
            )
            break

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
    )

    print("\n=== TRAINING COMPLETATO ===")
    print(f"Migliore epoca:   {best_epoch}")
    print(f"Migliore Micro-F1: {best_micro_f1:.4f}")
    print(f"Durata totale:     {total_seconds:.1f} s")
    print(f"Metadata:          {metadata_path}")


if __name__ == "__main__":
    main()