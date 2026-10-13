"""Caricamento della cache e costruzione dei batch PyTorch."""

import json
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache/preprocessed"))


def load_preprocessed_data():
    """Carica dataset e configurazione prodotti dal preprocessing."""

    dataset = load_from_disk(str(CACHE_DIR / "dataset"))

    with (CACHE_DIR / "preprocessing_config.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    return dataset, config


def make_collate_fn(pad_token_id):
    """Crea la funzione che assembla esempi di lunghezza diversa in un batch."""

    def collate_batch(examples):
        # Ogni notizia ha una lunghezza diversa: convertiamo prima
        # le singole sequenze in tensori indipendenti.
        sequences = [
            torch.tensor(example["input_ids"], dtype=torch.long)
            for example in examples
        ]

        # Il padding viene applicato solo fino alla sequenza più lunga
        # presente nel batch, evitando calcolo inutile.
        input_ids = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=pad_token_id,
        )

        # La mask vale 1 sui token reali e 0 sulle posizioni di padding.
        # Servirà al modello per escludere i PAD dal mean pooling.
        attention_mask = input_ids.ne(pad_token_id)

        # BCEWithLogitsLoss richiede target floating point.
        labels = torch.tensor(
            [example["labels"] for example in examples],
            dtype=torch.float32,
        )

        news_ids = [
            example["news_id"]
            for example in examples
        ]

        return {
            "news_id": news_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_batch


def main():
    """Smoke test della pipeline di caricamento e batching."""

    dataset, config = load_preprocessed_data()

    collate_fn = make_collate_fn(
        config["pad_token_id"]
    )

    # Usiamo un batch molto piccolo perché qui vogliamo soltanto
    # verificare la struttura dei tensori, non addestrare il modello.
    loader = DataLoader(
        dataset["train"],
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))

    print("\n=== TEST DATA PIPELINE ===")
    print("News ID:", batch["news_id"])
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("labels:", batch["labels"].shape)

    print(
        "Token reali per articolo:",
        batch["attention_mask"].sum(dim=1).tolist(),
    )


if __name__ == "__main__":
    main()