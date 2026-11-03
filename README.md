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

Inserisci la tua `OPENAI_API_KEY` in `.env` per attivare la chat. La chiave viene letta solo dal backend; modello e impostazioni RAG si scelgono nella pagina chat. Il primo utilizzo di un bugiardino AIFA lo scarica nella cache locale; per scaricare in anticipo tutti i PDF del catalogo esegui `python -m backend.leaflets`. I nuovi farmaci richiedono il caricamento del relativo PDF.

Le domande sulle quantità disponibili usano il tool `get_medicine_stock`: legge SQLite in sola lettura, distingue lotti scaduti e non scaduti e non richiede un filtro farmaco. Le domande cliniche continuano a usare il RAG dei bugiardini.

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
