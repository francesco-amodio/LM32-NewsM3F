FROM python:3.14-slim

# Evita file .pyc e rende immediatamente visibili i log Python.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Alcune librerie numeriche usate da PyTorch/scikit-learn
# richiedono il runtime OpenMP.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiamo prima le dipendenze per sfruttare la cache dei layer Docker:
# se cambia solo il codice, non occorre reinstallare tutti i pacchetti.
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# Il codice e la configurazione del progetto entrano nell'immagine.
# Dataset/cache e output verranno invece montati come volumi.
COPY src ./src
COPY config ./config
COPY README.md .

CMD ["python", "-m", "src.ddp_network_test", "--help"]