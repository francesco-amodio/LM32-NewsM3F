"""Preparazione iniziale del dataset News-M3F."""

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

    print("\nPrime 10 categorie selezionate:")
    for label in selected_labels[:10]:
        print(f"  {label}: {label_counts[label]:,}")

    print("\nSeed dello split:", RANDOM_SEED)


if __name__ == "__main__":
    main()