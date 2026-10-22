LM32-NewsM3F

Progetto individuale per il corso di Intelligenza Artificiale Scalabile.

Il progetto realizza un sistema distribuito per la classificazione multi-label di notizie del dataset News-M3F, utilizzando un modello PyTorch addestrato da zero.

L'esecuzione distribuita utilizza due nodi CPU, PyTorch Distributed Data Parallel (DDP) con backend Gloo e container Docker.



1. Requisiti
* Python 3.11
* Docker
* due nodi collegati in rete
* ambiente Linux/WSL per l'esecuzione distribuita

Le dipendenze Python sono definite in "requirements.txt".

Per installarle:

"python -m pip install -r requirements.txt"



2\. Preprocessing e cache

Il preprocessing viene eseguito una sola volta:

"python -m src.preprocess"

Lo script:

* scarica il dataset "News-M3F/News-M3F";
* crea gli split di training, validation e test;
* seleziona le classi Level2 con almeno 500 occorrenze;
* costruisce il vocabolario sul solo training;
* tokenizza e codifica testi e label;
* salva gli artefatti preprocessati in "cache/preprocessed/".

La directory "cache/preprocessed" deve essere copiata anche sul secondo nodo mantenendo lo stesso percorso relativo al repository.



3\. Configurazione dei nodi

La configurazione distribuita è contenuta in "config/distributed.json".

Prima dell'esecuzione verificare in particolare:

* "master\_addr": indirizzo IP del nodo rank 0;
* "world\_size": numero di nodi;
* hostname e rank dei nodi;
* interfaccia di rete utilizzata da Gloo;
* batch size locale;
* numero di epoche.

Nella configurazione utilizzata per il progetto:

HP-NOTEBOOK = rank 0
PC-Fisso = rank 1
world size = 2
backend = Gloo



4\. Build dell'immagine Docker

Su entrambi i nodi, dalla directory principale del repository:

"docker build -t lm32-newsm3f:compat ."

Il training distribuito viene eseguito all'interno dei container Docker.



5\. Avvio del workflow distribuito

Aprire WSL su entrambi i nodi e posizionarsi nella directory del progetto.

Avviare prima il nodo rank 0:

"python3 -m src.launch\_distributed"

Poi avviare lo stesso comando sul nodo rank 1:

"python3 -m src.launch\_distributed"

Il rank 0 genera il nome del run e lo comunica al secondo nodo.

Ogni esecuzione viene salvata automaticamente in una directory del tipo:

"outputs/AAAA.MM.GG\_OO.MM.SS/"

È possibile aggiungere una descrizione:

"python3 -m src.launch\_distributed --run-description 'run definitivo'"

ottenendo, ad esempio:

"outputs/2026.10.22\_17.36.36 - run definitivo/"

Per modificare il numero di epoche senza cambiare la configurazione:

"python3 -m src.launch\_distributed --epochs 1"



6\. Workflow

Il launcher esegue in sequenza tre candidati:

* A: Binary Cross-Entropy (BCE) non pesata;
* B: Binary Cross-Entropy (BCE) con "pos\_weight";
* C: Binary Cross-Entropy (BCE) con pesatura attenuata tramite radice quadrata.

Ogni candidato viene addestrato mediante DDP sui due nodi.

Al termine dei tre training, il rank 0 esegue il fan-in, seleziona il modello con il migliore Micro-F1 sulla validation e valuta automaticamente il checkpoint selezionato sul test set.

Gli output del run contengono:

"candidate\_A/"
"candidate\_B/"
"candidate\_C/"
"selection/"
"test/"



7\. Riproducibilità

Il progetto utilizza seed deterministici per le operazioni rilevanti. Dataset preprocessato, vocabolario e configurazione delle classi vengono memorizzati nella cache e riutilizzati dai due nodi.

I risultati sperimentali e la descrizione delle scelte architetturali sono riportati nella relazione del progetto.

