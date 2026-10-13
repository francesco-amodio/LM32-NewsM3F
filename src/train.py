"""Training single-process e validazione del classificatore News-M3F."""

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

# Soglia iniziale utilizzata per trasformare le probabilità
# sigmoid in predizioni multi-label binarie.
PREDICTION_THRESHOLD = 0.50

# Per ora manteniamo il training breve:
# vogliamo verificare correttamente anche la fase di validazione.
SMOKE_TEST_BATCHES = 100


def set_random_seed(seed):
    """Fissa i principali generatori casuali per favorire la riproducibilità."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_smoke_test(model, loader, optimizer, loss_function):
    """Esegue un breve training per verificare l'intera pipeline."""

    model.train()

    total_loss = 0.0

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

        total_loss += loss.item()

        if batch_index % 20 == 0:
            average_loss = total_loss / batch_index

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

    # eval() disattiva eventuali comportamenti specifici del training.
    # Nel nostro modello attuale non abbiamo dropout o batch normalization,
    # ma è buona pratica rendere esplicita la modalità di valutazione.
    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_targets = []
    all_predictions = []

    # Durante la validation non servono gradienti:
    # disabilitarli riduce memoria e calcolo.
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

            # Pesiamo la loss per il numero di esempi del batch,
            # così anche l'ultimo batch più piccolo viene trattato correttamente.
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= threshold
            ).to(torch.int32)

            all_predictions.append(
                predictions.cpu()
            )

            all_targets.append(
                batch["labels"].to(torch.int32).cpu()
            )

    targets = torch.cat(all_targets).numpy()
    predictions = torch.cat(all_predictions).numpy()

    validation_loss = total_loss / total_examples

    # Micro-F1 aggrega TP, FP e FN di tutte le classi:
    # sarà la metrica principale per confrontare i candidati.
    micro_f1 = f1_score(
        targets,
        predictions,
        average="micro",
        zero_division=0,
    )

    # Macro-F1 calcola prima l'F1 di ciascuna classe e poi
    # assegna lo stesso peso anche alle categorie meno frequenti.
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

    # Questi contatori sono diagnostici: se il modello prevede
    # sempre zero, ci permettono di capirlo immediatamente.
    predicted_positives = int(predictions.sum())
    actual_positives = int(targets.sum())

    return {
        "validation_loss": validation_loss,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "predicted_positives": predicted_positives,
        "actual_positives": actual_positives,
    }


def main():
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

        # La validation non deve essere casuale:
        # vogliamo semplicemente valutarne tutti gli esempi.
        shuffle=False,

        collate_fn=collate_fn,
        num_workers=0,
    )

    model = NewsClassifier(
        vocabulary_size=config["vocabulary_size"],
        num_classes=config["num_classes"],
        pad_token_id=config["pad_token_id"],
    )

    # Candidato A: loss BCE non pesata.
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("\n=== SMOKE TEST TRAINING ===")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch da eseguire: {SMOKE_TEST_BATCHES}")

    average_loss = train_smoke_test(
        model,
        train_loader,
        optimizer,
        loss_function,
    )

    print("\nTraining completato senza errori.")
    print(f"Loss media training: {average_loss:.4f}")

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