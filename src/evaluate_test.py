"""Valutazione finale sul test set del modello selezionato dal Fan-in."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Le CPU Ivy Bridge dei due nodi hanno mostrato instabilità con alcuni
# percorsi multithread di libtorch. Manteniamo anche nella valutazione
# finale la stessa configurazione stabile usata per il training DDP.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from src.data import load_preprocessed_data, make_collate_fn
from src.model import NewsClassifier
from src.train import (
    PREDICTION_THRESHOLD,
    VALIDATION_BATCH_SIZE,
    build_loss_function,
    evaluate_model,
)


DEFAULT_INPUT_ROOT = Path("outputs/distributed")


def parse_args():
    """Legge la directory contenente il risultato del workflow."""

    parser = argparse.ArgumentParser(
        description=(
            "Valuta sul test set il modello selezionato "
            "dal Fan-in dei candidati DDP."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=(
            "Directory contenente candidate_A/B/C e selection. "
            "Default: outputs/distributed"
        ),
    )

    return parser.parse_args()


def load_selection(input_root):
    """Carica e controlla il risultato prodotto dal Fan-in."""

    selection_path = (
        input_root
        / "selection"
        / "best_model.json"
    )

    if not selection_path.exists():
        raise FileNotFoundError(
            f"Risultato del Fan-in non trovato: {selection_path}"
        )

    with selection_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        selection = json.load(file)

    required_fields = (
        "execution_mode",
        "selected_candidate",
        "selected_score",
        "selected_epoch",
        "selected_checkpoint",
    )

    for field in required_fields:
        if field not in selection:
            raise ValueError(
                f"Campo '{field}' mancante in {selection_path}."
            )

    # Il test finale deve utilizzare esclusivamente un modello
    # proveniente dal workflow distribuito che abbiamo validato.
    if selection["execution_mode"] != "ddp":
        raise ValueError(
            "Il modello selezionato non proviene "
            "da un training DDP."
        )

    return selection_path, selection


def main():
    args = parse_args()

    selection_path, selection = load_selection(
        args.input_root
    )

    candidate = selection["selected_candidate"]
    selected_epoch = selection["selected_epoch"]

    checkpoint_path = Path(
        selection["selected_checkpoint"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint selezionato non trovato: {checkpoint_path}"
        )

    print("\n=== TEST FINALE ===")
    print(f"Selezione:          {selection_path}")
    print(f"Candidato:          {candidate}")
    print(f"Epoca selezionata:  {selected_epoch}")
    print(
        "Micro-F1 validation: "
        f"{selection['selected_score']:.4f}"
    )
    print(f"Checkpoint:         {checkpoint_path}")

    # Il preprocessing è già stato eseguito e memorizzato nella cache.
    # Qui utilizziamo il test set soltanto dopo che il Fan-in ha scelto
    # il modello migliore sulla validation.
    dataset, config = load_preprocessed_data()

    if "test" not in dataset:
        raise ValueError(
            "Lo split 'test' non è presente nel dataset preprocessato."
        )

    collate_fn = make_collate_fn(
        config["pad_token_id"]
    )

    test_loader = DataLoader(
        dataset["test"],
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Ricostruiamo esattamente la stessa architettura usata durante
    # il training prima di caricare i pesi del checkpoint.
    model = NewsClassifier(
        vocabulary_size=config["vocabulary_size"],
        num_classes=config["num_classes"],
        pad_token_id=config["pad_token_id"],
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # Questi controlli evitano di valutare per errore un checkpoint
    # diverso da quello scelto dal Fan-in.
    if checkpoint.get("candidate") != candidate:
        raise ValueError(
            "Il candidato dichiarato nel checkpoint "
            "non coincide con quello selezionato."
        )

    if checkpoint.get("epoch") != selected_epoch:
        raise ValueError(
            "L'epoca del checkpoint non coincide "
            "con quella selezionata dal Fan-in."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # La loss deve essere la stessa usata dal candidato selezionato.
    # Per B e C i pesi di classe vengono quindi ricostruiti usando
    # esclusivamente il training set, non il test set.
    loss_function, _ = build_loss_function(
        candidate,
        dataset["train"],
        config["num_classes"],
    )

    # evaluate_model è la stessa funzione usata per la validation.
    # Cambia soltanto il DataLoader: qui contiene dataset["test"].
    metrics = evaluate_model(
        model,
        test_loader,
        loss_function,
        threshold=PREDICTION_THRESHOLD,
    )

    # La funzione condivisa chiama questa quantità "validation_loss".
    # Nel risultato finale la rinominiamo per evitare ambiguità.
    test_loss = metrics.pop("validation_loss")

    test_metrics = {
        "test_loss": test_loss,
        **metrics,
    }

    result = {
        "execution_mode": "ddp",
        "evaluation_split": "test",
        "selected_candidate": candidate,
        "selected_epoch": selected_epoch,
        "validation_micro_f1": selection["selected_score"],
        "selected_checkpoint": str(checkpoint_path),
        "test_examples": len(dataset["test"]),
        "prediction_threshold": PREDICTION_THRESHOLD,
        "metrics": test_metrics,
    }

    output_dir = (
        args.input_root
        / "test"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "final_test.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Test examples:      {len(dataset['test']):,}")
    print(f"Test loss:          {test_loss:.4f}")
    print(
        f"Micro-F1:           "
        f"{test_metrics['micro_f1']:.4f}"
    )
    print(
        f"Macro-F1:           "
        f"{test_metrics['macro_f1']:.4f}"
    )
    print(
        f"Precision:          "
        f"{test_metrics['precision']:.4f}"
    )
    print(
        f"Recall:             "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        "\nRisultato test finale salvato in: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()