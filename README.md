# iFarmaco

Applicazione locale con interfaccia HTML/CSS/JavaScript, backend Python, database SQLite e chat RAG sui bugiardini.

## Avvio

Richiede Python 3.10 o successivo. Al primo avvio installa le dipendenze e crea `.env`:

Dalla cartella del progetto:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python run.py
```

Prima del blocco precedente, crea l'ambiente virtuale con `python -m venv .venv`.

Apri <http://127.0.0.1:8000>. Al primo avvio viene creato `data/ifarmaco.sqlite3` e i CSV in `data/` vengono importati una sola volta. Le modifiche successive sono salvate in SQLite; i CSV restano i file di origine e non vengono riscritti dall'applicazione.

Inserisci la tua `OPENAI_API_KEY` in `.env` per attivare la chat. Facoltativamente, imposta anche `COHERE_API_KEY` (la variabile vuota è già nel template `.env.example`) per abilitare **Cohere Rerank** nelle impostazioni. Senza questa chiave la chat funziona normalmente e l'opzione Cohere non viene mostrata. Quando si usa Cohere, la domanda e i passaggi candidati vengono inviati al servizio Cohere per il riordino. Le chiavi sono lette solo dal backend; modello di risposta, modello di embedding e impostazioni RAG si configurano nella pagina Impostazioni e vengono salvati in SQLite, condivisi da questa installazione locale. Per partire vengono selezionati `gpt-4o-mini`, `text-embedding-3-small`, ricerca semantica, nessun reranker, 4 fonti, memoria buffer e budget 700 token. La ricerca ibrida combina la ricerca semantica con BM25 e Reciprocal Rank Fusion (RRF); il suo indice lessicale viene costruito in memoria al primo uso e aggiornato quando si indicizzano nuovi documenti. Il reranker recupera fino a 50 candidati e usa Cohere per riordinarli prima di passare al generatore i primi *k*. All'avvio, con la chiave OpenAI configurata, i bugiardini del catalogo vengono scaricati se necessario e indicizzati con il modello di embedding selezionato; gli indici già presenti vengono riutilizzati. Anche ogni nuovo bugiardino viene indicizzato subito dopo il caricamento. Il primo avvio può richiedere tempo e comportare costi API; gli errori di download o parsing sono riportati nel terminale. Cambiare modello di embedding crea un indice separato, mantenendo riutilizzabile quello esistente. Le opzioni di visualizzazione possono mostrare una sintesi verificabile delle fonti (non il ragionamento interno del modello) e una stima del costo OpenAI per interazione in USD; questa stima non include eventuali costi Cohere, e l'importo effettivo può variare per caching, prezzi aggiornati o condizioni dell'account. Le tariffe di stima sono definite in `backend/chat_service.py` e vanno confrontate con i [prezzi API correnti](https://developers.openai.com/api/docs/pricing). Il filtro del farmaco resta facoltativo nella chat. Per scaricare soltanto i PDF senza indicizzarli esegui `python -m backend.leaflets`. I nuovi farmaci richiedono il caricamento del relativo PDF.

Le domande sulle quantità disponibili usano il tool `get_medicine_stock`: legge in sola lettura le tabelle `farmaci` e `scorte`, distingue lotti scaduti e non scaduti e, se non viene indicato un farmaco, interroga tutte le giacenze ordinandole per quantità non scaduta. Le domande cliniche continuano a usare il RAG dei bugiardini; per ogni fonte la risposta mostra anche un breve estratto del testo recuperato, oltre al link alla pagina del PDF.

PDF, indici vettoriali e sessioni sono in `data/` e sono esclusi da Git: restano nell'installazione locale senza essere aggiunti alla repository versionata.

Per arrestare il server premi `Ctrl+C`.

## Struttura

```text
backend/       Database, repository, API HTTP, RAG e avvio server
data/          CSV iniziali, database e storage locale ignorato da Git
frontend/      Interfaccia HTML, CSS e JavaScript
tests/         Test automatici
run.py         Punto d'ingresso dell'applicazione
requirements.txt Dipendenze Python
```

## Funzionalità dati

- Consultazione e ricerca di farmaci e lotti di scorta.
- Creazione e modifica dei farmaci e delle scorte.
- Eliminazione dei lotti di scorta.
- ID dei farmaci trattati come testo per conservare gli zeri iniziali.
- Ogni lotto deve riferirsi a un farmaco esistente; quantità non negative e date `AAAA-MM-GG`.

## Test

```powershell
python -m unittest discover -s tests -v
```

## Valutazione del RAG

Con `OPENAI_API_KEY` configurata, esegui il banco di prova sui bugiardini:

```powershell
python -m backend.rag_evaluation --top-k 4
python -m backend.rag_evaluation --top-k 4 --retrieval-mode hybrid
```

Il comando usa un foglietto del catalogo, indicizza il PDF se necessario e
interroga il vero indice. Misura `Recall@k` e `MRR@k` su tre domande
con passaggio e pagina attesi; il caso sul prezzo, assente dal foglietto, è una
verifica manuale del rifiuto. Il report mostra anche la posizione dei passaggi
fra i candidati e i token effettivi del contesto recuperato. Cambia `--top-k`
per confrontare, per esempio, `1`, `4` e `8`. Le chiamate di embedding possono
avere un costo API; PDF e indice rimangono nello storage locale.

Per provare anche la generazione, il grounding e il fenomeno
lost-in-the-middle:

```powershell
python -m backend.rag_evaluation --top-k 4 --generate
```

Questa modalità effettua chiamate al modello di risposta, ripete la domanda
con il passaggio atteso all'inizio, al centro e alla fine del medesimo contesto,
e stampa le risposte per il controllo umano. Verifica che il fatto sia
supportato dai passaggi mostrati, che la pagina citata sia corretta e che la
domanda sul prezzo non riceva una risposta inventata. La qualità fattuale e la
sensibilità alla posizione non si possono certificare con un test automatico
basato soltanto sulla presenza di una frase.

Il budget memoria configurato nell'app limita la cronologia della chat, non i
chunk recuperati: confronta separatamente i token del contesto riportati dal
banco di prova. Per una verifica ripetibile, annota modello, `top-k`, Recall,
MRR e risultato delle tre prove di generazione; dopo una modifica confrontali
con la stessa configurazione e gli stessi casi.
