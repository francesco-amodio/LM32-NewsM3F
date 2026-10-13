"""Modello neurale per la classificazione multi-label di News-M3F."""

import torch
from torch import nn

from src.data import load_preprocessed_data, make_collate_fn
from torch.utils.data import DataLoader


EMBEDDING_DIM = 64


class NewsClassifier(nn.Module):
    """Classificatore testuale Embedding -> Mean Pooling -> Linear."""

    def __init__(
        self,
        vocabulary_size,
        num_classes,
        pad_token_id,
        embedding_dim=EMBEDDING_DIM,
    ):
        super().__init__()

        # Ogni token del vocabolario viene rappresentato tramite
        # un vettore appreso durante il training.
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )

        # Il classificatore produce un logit indipendente
        # per ciascuna delle classi Level2 selezionate.
        self.classifier = nn.Linear(
            embedding_dim,
            num_classes,
        )

    def forward(self, input_ids, attention_mask):
        """Calcola i logits multi-label per un batch di notizie."""

        embeddings = self.embedding(input_ids)

        # La mask viene estesa con una dimensione finale per poterla
        # applicare ai vettori embedding di ogni token.
        mask = attention_mask.unsqueeze(-1).to(
            embeddings.dtype
        )

        # Escludiamo i token PAD dalla somma degli embedding.
        masked_embeddings = embeddings * mask
        summed_embeddings = masked_embeddings.sum(dim=1)

        # Numero di token reali presenti in ogni notizia.
        token_counts = mask.sum(dim=1).clamp(min=1.0)

        # Mean pooling: otteniamo un solo vettore rappresentativo
        # dell'intera notizia facendo la media dei token reali.
        pooled = summed_embeddings / token_counts

        # Non applichiamo sigmoid qui:
        # BCEWithLogitsLoss lavorerà direttamente sui logits.
        logits = self.classifier(pooled)

        return logits


def main():
    """Smoke test del modello su un piccolo batch reale."""

    dataset, config = load_preprocessed_data()

    collate_fn = make_collate_fn(
        config["pad_token_id"]
    )

    loader = DataLoader(
        dataset["train"],
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))

    model = NewsClassifier(
        vocabulary_size=config["vocabulary_size"],
        num_classes=config["num_classes"],
        pad_token_id=config["pad_token_id"],
    )

    logits = model(
        batch["input_ids"],
        batch["attention_mask"],
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\n=== TEST MODELLO ===")
    print(model)
    print(f"\nNumero parametri: {total_parameters:,}")
    print("Forma input:", batch["input_ids"].shape)
    print("Forma output:", logits.shape)


if __name__ == "__main__":
    main()