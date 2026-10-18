"""Fan-in dei risultati DDP e selezione del miglior modello candidato."""

import argparse
import json
from pathlib import Path


CANDIDATES = ("A", "B", "C")

# La Micro-F1 sul validation set è la metrica primaria
# stabilita per confrontare i candidati.
SELECTION_METRIC = "best_micro_f1"


def parse_args():
    """Legge i parametri del Fan-in dalla riga di comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Confronta i candidati A/B/C prodotti dal training DDP "
            "e seleziona quello con la Micro-F1 migliore."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/distributed"),
        help=(
            "Directory che contiene candidate_A, candidate_B e candidate_C. "
            "Default: outputs/distributed"
        ),
    )

    return parser.parse_args()


def load_candidate_metadata(candidate, input_root):
    """Carica e controlla i metadata DDP prodotti da un candidato."""

    metadata_path = (
        input_root
        / f"candidate_{candidate}"
        / "metadata"
        / f"candidate_{candidate}.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata DDP non trovati per il candidato {candidate}: "
            f"{metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    # Questi campi sono necessari per eseguire il Fan-in
    # e ricostruire le metriche dell'epoca migliore.
    required_fields = (
        "candidate",
        "execution_mode",
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

    # Impediamo di mescolare accidentalmente risultati
    # provenienti da modalità di training differenti.
    if metadata["execution_mode"] != "ddp":
        raise ValueError(
            f"Il candidato {candidate} non proviene da un training DDP: "
            f"execution_mode={metadata['execution_mode']}"
        )

    if metadata["candidate"] != candidate:
        raise ValueError(
            f"Il file {metadata_path} dichiara il candidato "
            f"{metadata['candidate']} invece di {candidate}."
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
    args = parse_args()

    input_root = args.input_root
    selection_dir = input_root / "selection"

    candidate_results = []

    print(
        f"\nRoot risultati DDP: {input_root}"
    )

    # FAN-IN:
    # raccogliamo in un unico punto i risultati dei tre rami
    # indipendenti prodotti dal Fan-out.
    for candidate in CANDIDATES:
        metadata_path, metadata = load_candidate_metadata(
            candidate,
            input_root,
        )

        best_epoch_result = get_best_epoch_result(
            metadata
        )

        candidate_results.append(
            {
                "candidate": candidate,
                "execution_mode": metadata["execution_mode"],
                "metadata_path": str(metadata_path),
                "best_checkpoint": metadata["best_checkpoint"],
                "best_epoch": metadata["best_epoch"],
                "micro_f1": metadata["best_micro_f1"],
                "macro_f1": best_epoch_result["macro_f1"],
                "precision": best_epoch_result["precision"],
                "recall": best_epoch_result["recall"],
            }
        )

    # FAN-IN:
    # i tre rami convergono qui e viene selezionato
    # il candidato con la Micro-F1 maggiore sul validation set.
    best_result = max(
        candidate_results,
        key=lambda result: result["micro_f1"],
    )

    selection_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection = {
        "execution_mode": "ddp",
        "selection_metric": SELECTION_METRIC,
        "input_root": str(input_root),
        "selected_candidate": best_result["candidate"],
        "selected_score": best_result["micro_f1"],
        "selected_epoch": best_result["best_epoch"],
        "selected_checkpoint": best_result["best_checkpoint"],
        "candidates": candidate_results,
    }

    selection_path = (
        selection_dir / "best_model.json"
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

    print("\n=== FAN-IN DDP: CONFRONTO CANDIDATI ===")

    for result in candidate_results:
        print(
            f"Candidato {result['candidate']} | "
            f"Micro-F1: {result['micro_f1']:.4f} | "
            f"Macro-F1: {result['macro_f1']:.4f} | "
            f"Precision: {result['precision']:.4f} | "
            f"Recall: {result['recall']:.4f}"
        )

    print("\n=== MODELLO DDP SELEZIONATO ===")
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