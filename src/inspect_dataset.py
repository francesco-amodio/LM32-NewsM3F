"""Controllo preliminare del dataset News-M3F."""

from datasets import load_dataset


DATASET_NAME = "News-M3F/News-M3F"


def main():
    # Carichiamo il dataset direttamente da Hugging Face.
    # In questa fase ci interessa verificare disponibilità e struttura,
    # non eseguire ancora alcun preprocessing.
    dataset = load_dataset(DATASET_NAME)

    print("\n=== NEWS-M3F ===")
    print(dataset)

    for split_name, split_data in dataset.items():
        print(f"\nSplit: {split_name}")
        print(f"Numero di esempi: {len(split_data):,}")
        print(f"Colonne: {split_data.column_names}")

    # Visualizziamo un solo record per verificare il formato effettivo
    # delle feature testuali e delle label.
    print("\n=== PRIMO RECORD DEL TRAINING SET ===")
    print(dataset["train"][0])


if __name__ == "__main__":
    main()