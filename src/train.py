"""Training single-process e validazione del classificatore News-M3F."""

import argparse
import math
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data import load_preprocessed_data, make_collate_fn
from src.model import NewsClassifier


RANDOM_SEED = 42

BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 128

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PREDICTION_THRESHOLD = 0.50

# Per ora eseguiamo un test breve.
SMOKE_TEST_BATCHES = 100


def parse_args():
    """Legge da riga di comando la configurazione candidata da testare."""

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

    return parser.parse_args()


def set_random_seed(seed):
    """Fissa i principali generatori casuali per la riproducibilità."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_positive_counts(train_dataset, num_classes):
    """Conta gli esempi positivi per ciascuna delle classi."""

    positive_counts = np.zeros(
        num_classes,
        dtype=np.int64,
    )

    # Le label sono vettori binari di 90 elementi.
    # Sommando i vettori otteniamo il numero di positivi per classe.
    for labels in train_dataset["labels"]:
        positive_counts += np.asarray(
            labels,
            dtype=np.int64,
        )

    return positive_counts


def build_loss_function(candidate, train_dataset, num_classes):
    """Costruisce la loss corrispondente al candidato A, B o C."""

    if candidate == "A":
        return nn.BCEWithLogitsLoss(), None

    positive_counts = compute_positive_counts(
        train_dataset,
        num_classes,
    )

    total_examples = len(train_dataset)
    negative_counts = total_examples - positive_counts

    # Evitiamo divisioni per zero; nel nostro dataset le classi
    # selezionate hanno comunque tutte esempi positivi.
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
        # La radice quadrata attenua l'effetto dei pesi molto elevati,
        # mantenendo comunque maggiore attenzione sulle classi rare.
        weights = np.sqrt(full_weights).astype(
            np.float32
        )

    else:
        raise ValueError(
            f"Candidato non riconosciuto: {candidate}"
        )

    pos_weight = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    return (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        ),
        pos_weight,
    )


def train_smoke_test(
    model,
    loader,
    optimizer,
    loss_function,
):
    """Esegue un breve training per verificare l'intera pipeline."""

    model.train()
    total_loss = 0.0

    for batch_index, batch in enumerate(
        loader,
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

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_index % 20 == 0:
            average_loss = (
                total_loss / batch_index
            )

            print(
                f"Batch {batch_index:3d} | "
                f"Loss corrente: {loss.item():.4f} | "
                f"Loss media: {average_loss:.4f}"
            )

        if batch_index >= SMOKE_TEST_BATCHES:
            break

    return total_loss / batch_index


def evaluate_model(
    model,
    loader,
    loss_function,
    threshold=PREDICTION_THRESHOLD,
):
    """Valuta il modello senza modificare i parametri."""

    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_targets = []
    all_predictions = []

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

            total_loss += (
                loss.item() * batch_size
            )
            total_examples += batch_size

            probabilities = torch.sigmoid(
                logits
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

    targets = torch.cat(
        all_targets
    ).numpy()

    predictions = torch.cat(
        all_predictions
    ).numpy()

    validation_loss = (
        total_loss / total_examples
    )

    micro_f1 = f1_score(
        targets,
        predictions,
        average="micro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    precision = precision_score(
        targets,
        predictions,
        average="micro",
        zero_division=0,
    )

    recall = recall_score(
        targets,
        predictions,
        average="micro",
        zero_division=0,
    )

    return {
        "validation_loss": validation_loss,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "predicted_positives": int(
            predictions.sum()
        ),
        "actual_positives": int(
            targets.sum()
        ),
    }


def main():
    args = parse_args()

    set_random_seed(RANDOM_SEED)

    dataset, config = load_preprocessed_data()

    collate_fn = make_collate_fn(
        config["pad_token_id"]
    )

    train_loader = DataLoader(
        dataset["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
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

    loss_function, pos_weight = (
        build_loss_function(
            args.candidate,
            dataset["train"],
            config["num_classes"],
        )
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("\n=== CONFIGURAZIONE ===")
    print(f"Candidato: {args.candidate}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    if pos_weight is None:
        print("pos_weight: nessuno")
    else:
        print(
            f"pos_weight min: "
            f"{pos_weight.min().item():.2f}"
        )
        print(
            f"pos_weight max: "
            f"{pos_weight.max().item():.2f}"
        )
        print(
            f"pos_weight medio: "
            f"{pos_weight.mean().item():.2f}"
        )

    print(
        f"Batch da eseguire: "
        f"{SMOKE_TEST_BATCHES}"
    )

    print("\n=== TRAINING ===")

    average_loss = train_smoke_test(
        model,
        train_loader,
        optimizer,
        loss_function,
    )

    print(
        f"\nLoss media training: "
        f"{average_loss:.4f}"
    )

    print("\n=== VALIDATION ===")

    metrics = evaluate_model(
        model,
        validation_loader,
        loss_function,
    )

    print(
        f"Validation loss:      "
        f"{metrics['validation_loss']:.4f}"
    )
    print(
        f"Micro-F1:             "
        f"{metrics['micro_f1']:.4f}"
    )
    print(
        f"Macro-F1:             "
        f"{metrics['macro_f1']:.4f}"
    )
    print(
        f"Precision (micro):     "
        f"{metrics['precision']:.4f}"
    )
    print(
        f"Recall (micro):        "
        f"{metrics['recall']:.4f}"
    )
    print(
        f"Predizioni positive:   "
        f"{metrics['predicted_positives']:,}"
    )
    print(
        f"Label positive reali:  "
        f"{metrics['actual_positives']:,}"
    )


if __name__ == "__main__":
    main()