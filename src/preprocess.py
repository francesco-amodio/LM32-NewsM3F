"""Preparazione iniziale del dataset News-M3F."""

import re
from collections import Counter

from datasets import load_dataset


DATASET_NAME = "News-M3F/News-M3F"

# Seed fisso per rendere riproducibile la divisione train/validation.
RANDOM_SEED = 42

# Una categoria Level2 viene mantenuta solo se compare almeno
# questo numero di volte nel training set ufficiale.
MIN_LABEL_FREQUENCY = 500

# Percentuale del training ufficiale riservata alla validazione.
VALIDATION_SIZE = 0.10

# Numero massimo di token reali mantenuti nel vocabolario.
# PAD e UNK verranno aggiunti separatamente.
MAX_VOCAB_TOKENS = 30_000

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def select_level2_labels(train_dataset):
    """Seleziona le categorie Level2 sufficientemente rappresentate."""

    label_counts = Counter()

    # Ogni notizia può avere più label Level2.
    # Counter aggiorna quindi le frequenze usando tutte le label presenti.
    for labels in train_dataset["label_level2"]:
        label_counts.update(labels)

    selected_labels = sorted(
        label
        for label, count in label_counts.items()
        if count >= MIN_LABEL_FREQUENCY
    )

    return selected_labels, label_counts


def normalize_text(text):
    """Normalizza il testo prima della tokenizzazione."""

    # Uniformiamo maiuscole/minuscole e apostrofi tipografici.
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'")

    # Sequenze di spazi, tab e ritorni a capo vengono ridotte
    # a un singolo spazio.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_text(text):
    """Trasforma una stringa normalizzata in una lista di token."""

    # Manteniamo parole e numeri; gli apostrofi interni vengono
    # conservati per token come "don't" o "it's".
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)


def build_vocabulary(train_dataset):
    """Costruisce il vocabolario usando esclusivamente il training effettivo."""

    token_counts = Counter()

    # Il conteggio viene eseguito solo sul training effettivo:
    # validation e test non contribuiscono alla scelta del vocabolario.
    for text in train_dataset["text"]:
        normalized_text = normalize_text(text)
        tokens = tokenize_text(normalized_text)
        token_counts.update(tokens)

    most_common_tokens = token_counts.most_common(MAX_VOCAB_TOKENS)

    # PAD=0 e UNK=1 rimangono fissi.
    # I token reali iniziano dall'indice 2.
    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token, _ in most_common_tokens:
        vocabulary[token] = len(vocabulary)

    return vocabulary, token_counts


def main():
    dataset = load_dataset(DATASET_NAME)

    official_train = dataset["train"]
    official_test = dataset["test"]

    # La scelta delle categorie viene effettuata esclusivamente
    # sul training ufficiale. Il test non influenza il task.
    selected_labels, label_counts = select_level2_labels(official_train)

    # La divisione è deterministica grazie al seed fisso:
    # esecuzioni successive produrranno gli stessi sottoinsiemi.
    split = official_train.train_test_split(
        test_size=VALIDATION_SIZE,
        seed=RANDOM_SEED,
    )

    train_dataset = split["train"]
    validation_dataset = split["test"]

    print("\n=== PREPARAZIONE NEWS-M3F ===")
    print(f"Training effettivo: {len(train_dataset):,}")
    print(f"Validation:         {len(validation_dataset):,}")
    print(f"Test ufficiale:     {len(official_test):,}")

    print(
        f"\nCategorie Level2 con frequenza >= "
        f"{MIN_LABEL_FREQUENCY}: {len(selected_labels)}"
    )

    print("\nCostruzione del vocabolario globale...")
    vocabulary, token_counts = build_vocabulary(train_dataset)

    print(f"Token distinti trovati: {len(token_counts):,}")
    print(f"Dimensione vocabolario: {len(vocabulary):,}")
    print(f"  {PAD_TOKEN}: {vocabulary[PAD_TOKEN]}")
    print(f"  {UNK_TOKEN}: {vocabulary[UNK_TOKEN]}")

    print("\n10 token più frequenti:")
    for token, count in token_counts.most_common(10):
        print(f"  {token}: {count:,}")

    print("\nSeed dello split:", RANDOM_SEED)


if __name__ == "__main__":
    main()