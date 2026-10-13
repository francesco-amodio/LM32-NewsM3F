"""Fan-in dei risultati e selezione del miglior modello candidato."""

import json
from pathlib import Path


OUTPUT_DIR = Path("outputs")
METADATA_DIR = OUTPUT_DIR / "metadata"
SELECTION_DIR = OUTPUT_DIR / "selection"

CANDIDATES = ("A", "B", "C")

# La Micro-F1 sul validation set è la metrica primaria
# stabilita per confrontare i candidati.
SELECTION_METRIC = "best_micro_f1"


def load_candidate_metadata(candidate):
    """Carica e controlla i metadata prodotti dal training di un candidato."""

    metadata_path = (
        METADATA_DIR
        / f"candidate_{candidate}.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata non trovati per il candidato {candidate}: "
            f"{metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    # Controlliamo subito i campi necessari al fan-in,
    # così eventuali output incompleti vengono rilevati chiaramente.
    required_fields = (
        "candidate",
        "best_micro_f1",
        "best_epoch",
        "best_checkpoint",
        "history",
    )

    for field in required_fields:
        if field not in metadata:
            raise ValueError(
                f"Campo '{field}' mancante nei metadata "
                f"del candidato {candidate}."
            )

    return metadata_path, metadata


def get_best_epoch_result(metadata):
    """Recupera le metriche complete dell'epoca selezionata."""

    best_epoch = metadata["best_epoch"]

    for epoch_result in metadata["history"]:
        if epoch_result["epoch"] == best_epoch:
            return epoch_result

    raise ValueError(
        f"Risultati dell'epoca migliore {best_epoch} "
        f"non trovati per il candidato {metadata['candidate']}."
    )


def main():
    candidate_results = []

    # FAN-IN: raccogliamo in un unico punto i risultati
    # prodotti indipendentemente dai candidati A, B e C.
    for candidate in CANDIDATES:
        metadata_path, metadata = load_candidate_metadata(
            candidate
        )

        best_epoch_result = get_best_epoch_result(
            metadata
        )

        candidate_results.append(
            {
                "candidate": candidate,
                "metadata_path": str(metadata_path),
                "best_checkpoint": metadata["best_checkpoint"],
                "best_epoch": metadata["best_epoch"],
                "micro_f1": metadata["best_micro_f1"],
                "macro_f1": best_epoch_result["macro_f1"],
                "precision": best_epoch_result["precision"],
                "recall": best_epoch_result["recall"],
            }
        )

    # Il criterio di selezione è esclusivamente la migliore
    # Micro-F1 ottenuta sul validation set.
    best_result = max(
        candidate_results,
        key=lambda result: result["micro_f1"],
    )

    SELECTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection = {
        "selection_metric": SELECTION_METRIC,
        "selected_candidate": best_result["candidate"],
        "selected_score": best_result["micro_f1"],
        "selected_epoch": best_result["best_epoch"],
        "selected_checkpoint": best_result["best_checkpoint"],
        "candidates": candidate_results,
    }

    selection_path = (
        SELECTION_DIR / "best_model.json"
    )

    with selection_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selection,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== FAN-IN: CONFRONTO CANDIDATI ===")

    for result in candidate_results:
        print(
            f"Candidato {result['candidate']} | "
            f"Micro-F1: {result['micro_f1']:.4f} | "
            f"Macro-F1: {result['macro_f1']:.4f} | "
            f"Precision: {result['precision']:.4f} | "
            f"Recall: {result['recall']:.4f}"
        )

    print("\n=== MODELLO SELEZIONATO ===")
    print(
        f"Candidato:  {best_result['candidate']}"
    )
    print(
        f"Epoca:      {best_result['best_epoch']}"
    )
    print(
        f"Micro-F1:   {best_result['micro_f1']:.4f}"
    )
    print(
        f"Checkpoint: {best_result['best_checkpoint']}"
    )
    print(
        f"Risultato fan-in salvato in: {selection_path}"
    )


if __name__ == "__main__":
    main()