"""Preprocessing e caching del dataset News-M3F."""

import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, load_dataset


DATASET_NAME = "News-M3F/News-M3F"

# Seed fisso per rendere riproducibile la divisione train/validation.
RANDOM_SEED = 42

# Una categoria Level2 viene mantenuta se compare almeno
# questo numero di volte nel training ufficiale.
MIN_LABEL_FREQUENCY = 500

# Percentuale del training ufficiale riservata alla validazione.
VALIDATION_SIZE = 0.10

# Numero massimo di token reali mantenuti nel vocabolario.
MAX_VOCAB_TOKENS = 30_000

# Lunghezza massima di una notizia dopo la tokenizzazione.
# Le sequenze più lunghe vengono troncate; il padding sarà
# effettuato successivamente durante la creazione dei batch.
MAX_SEQUENCE_LENGTH = 768

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Il percorso è configurabile tramite variabile d'ambiente:
# questa scelta ci faciliterà quando il codice verrà eseguito in Docker.
CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache/preprocessed"))


def select_level2_labels(train_dataset):
    """Seleziona le categorie Level2 sufficientemente rappresentate."""

    label_counts = Counter()

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

    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_text(text):
    """Trasforma una stringa normalizzata in una lista di token."""

    # Manteniamo parole e numeri e conserviamo gli apostrofi interni,
    # ad esempio in "don't" e "it's".
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)


def build_vocabulary(train_dataset):
    """Costruisce il vocabolario usando solo il training effettivo."""

    token_counts = Counter()

    for text in train_dataset["text"]:
        normalized_text = normalize_text(text)
        tokens = tokenize_text(normalized_text)
        token_counts.update(tokens)

    most_common_tokens = token_counts.most_common(MAX_VOCAB_TOKENS)

    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token, _ in most_common_tokens:
        vocabulary[token] = len(vocabulary)

    return vocabulary, token_counts


def encode_text(text, vocabulary):
    """Converte il testo in una sequenza di ID del vocabolario."""

    normalized_text = normalize_text(text)
    tokens = tokenize_text(normalized_text)

    # I token non presenti tra i 30.000 più frequenti vengono
    # rappresentati con l'ID di <UNK>.
    unk_id = vocabulary[UNK_TOKEN]
    token_ids = [
        vocabulary.get(token, unk_id)
        for token in tokens
    ]

    return token_ids[:MAX_SEQUENCE_LENGTH]


def encode_labels(labels, label_to_index):
    """Converte le label Level2 in un vettore multi-label binario."""

    target = [0] * len(label_to_index)

    # Ogni posizione del vettore corrisponde a una delle 90 classi.
    # Una notizia può quindi avere più posizioni impostate a 1.
    for label in labels:
        index = label_to_index.get(label)

        if index is not None:
            target[index] = 1

    return target


def preprocess_split(dataset, vocabulary, label_to_index):
    """Preprocessa uno split mantenendo solo i dati necessari al training."""

    def encode_example(example):
        return {
            # Manteniamo news_id per poter ricondurre eventuali
            # predizioni o errori al record originale.
            "news_id": example["news_id"],
            "input_ids": encode_text(
                example["text"],
                vocabulary,
            ),
            "labels": encode_labels(
                example["label_level2"],
                label_to_index,
            ),
        }

    return dataset.map(
        encode_example,
        remove_columns=dataset.column_names,
    )


def save_json(data, path):
    """Salva un artefatto JSON in formato leggibile."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main():
    dataset = load_dataset(DATASET_NAME)

    official_train = dataset["train"]
    official_test = dataset["test"]

    # Le classi del task vengono definite esclusivamente utilizzando
    # il training ufficiale; il test non influenza questa scelta.
    selected_labels, label_counts = select_level2_labels(
        official_train
    )

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

    label_to_index = {
        label: index
        for index, label in enumerate(selected_labels)
    }

    print("\nTokenizzazione e codifica del training...")
    processed_train = preprocess_split(
        train_dataset,
        vocabulary,
        label_to_index,
    )

    print("\nTokenizzazione e codifica della validation...")
    processed_validation = preprocess_split(
        validation_dataset,
        vocabulary,
        label_to_index,
    )

    print("\nTokenizzazione e codifica del test...")
    processed_test = preprocess_split(
        official_test,
        vocabulary,
        label_to_index,
    )

    processed_dataset = DatasetDict(
        {
            "train": processed_train,
            "validation": processed_validation,
            "test": processed_test,
        }
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_cache_path = CACHE_DIR / "dataset"

    # La cache è un artefatto rigenerabile: se esiste già,
    # viene sostituita in modo da evitare dati incoerenti
    # provenienti da una precedente configurazione.
    if dataset_cache_path.exists():
        shutil.rmtree(dataset_cache_path)

    processed_dataset.save_to_disk(
        str(dataset_cache_path)
    )

    save_json(
        vocabulary,
        CACHE_DIR / "vocabulary.json",
    )

    save_json(
        selected_labels,
        CACHE_DIR / "labels.json",
    )

    preprocessing_config = {
        "dataset": DATASET_NAME,
        "random_seed": RANDOM_SEED,
        "validation_size": VALIDATION_SIZE,
        "min_label_frequency": MIN_LABEL_FREQUENCY,
        "num_classes": len(selected_labels),
        "max_vocab_tokens": MAX_VOCAB_TOKENS,
        "vocabulary_size": len(vocabulary),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "pad_token_id": vocabulary[PAD_TOKEN],
        "unk_token_id": vocabulary[UNK_TOKEN],
    }

    save_json(
        preprocessing_config,
        CACHE_DIR / "preprocessing_config.json",
    )

    print("\n=== CACHE CREATA ===")
    print(f"Percorso: {CACHE_DIR.resolve()}")

    print(
        f"Training:   {len(processed_dataset['train']):,}"
    )
    print(
        f"Validation: {len(processed_dataset['validation']):,}"
    )
    print(
        f"Test:       {len(processed_dataset['test']):,}"
    )

    # Controllo rapido di un record già preprocessato.
    sample = processed_dataset["train"][0]

    print("\nEsempio preprocessato:")
    print(f"news_id: {sample['news_id']}")
    print(
        f"Numero token: {len(sample['input_ids'])}"
    )
    print(
        f"Primi 20 ID: {sample['input_ids'][:20]}"
    )
    print(
        f"Label positive: {sum(sample['labels'])}"
    )


if __name__ == "__main__":
    main()