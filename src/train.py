"""Training single-process del classificatore News-M3F."""

import random

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data import load_preprocessed_data, make_collate_fn
from src.model import NewsClassifier


RANDOM_SEED = 42

# Valori iniziali del training.
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Per il primo test non percorriamo l'intero dataset:
# ci interessa verificare che forward, loss, backward e optimizer funzionino.
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
        # A ogni iterazione eliminiamo i gradienti del batch precedente.
        optimizer.zero_grad()

        logits = model(
            batch["input_ids"],
            batch["attention_mask"],
        )

        loss = loss_function(
            logits,
            batch["labels"],
        )

        # Calcolo dei gradienti rispetto a tutti i parametri addestrabili.
        loss.backward()

        # AdamW utilizza i gradienti appena calcolati per aggiornare
        # embedding e classificatore.
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

        # Manteniamo inizialmente il caricamento single-process.
        # Il parallelismo distribuito verrà introdotto e verificato
        # separatamente, evitando di sommare subito più fonti di complessità.
        num_workers=0,
    )

    model = NewsClassifier(
        vocabulary_size=config["vocabulary_size"],
        num_classes=config["num_classes"],
        pad_token_id=config["pad_token_id"],
    )

    # Candidato A: BCE multi-label non pesata.
    # Le varianti B e C verranno aggiunte dopo aver verificato
    # il corretto funzionamento del training di base.
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
    print(f"Loss media finale: {average_loss:.4f}")


if __name__ == "__main__":
    main()