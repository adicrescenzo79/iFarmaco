import argparse
import re
import tempfile
import time
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from backend.database import Database

MAX_LEAFLET_SIZE = 25 * 1024 * 1024
LEAFLET_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")


class LeafletStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path_for(self, medicine_id: str) -> Path:
        if not LEAFLET_ID_PATTERN.fullmatch(medicine_id):
            raise ValueError("ID farmaco non valido.")
        return self.directory / f"{medicine_id}.pdf"

    def stage(self, medicine_id: str, content: bytes) -> Path:
        validate_pdf(content)
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pdf.tmp", dir=self.directory, delete=False
        ) as file:
            file.write(content)
            return Path(file.name)

    def commit(self, medicine_id: str, staged_path: Path) -> Path:
        destination = self.path_for(medicine_id)
        staged_path.replace(destination)
        return destination

    def save(self, medicine_id: str, content: bytes) -> Path:
        staged_path = self.stage(medicine_id, content)
        try:
            return self.commit(medicine_id, staged_path)
        finally:
            staged_path.unlink(missing_ok=True)

    def find(self, medicine_id: str) -> Path | None:
        path = self.path_for(medicine_id)
        return path if path.is_file() else None


def validate_pdf(content: bytes) -> None:
    if not isinstance(content, bytes) or len(content) < 8:
        raise ValueError("Carica un file PDF valido.")
    if len(content) > MAX_LEAFLET_SIZE:
        raise ValueError("Il PDF supera il limite di 25 MB.")
    if not content.startswith(b"%PDF-"):
        raise ValueError("Il file caricato non è un PDF valido.")
    try:
        from pypdf import PdfReader

        if not PdfReader(BytesIO(content)).pages:
            raise ValueError("Il PDF non contiene pagine.")
    except Exception as error:
        raise ValueError("Il file caricato non è un PDF leggibile.") from error


def fetch_aifa_pdf(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "api.aifa.gov.it":
        raise ValueError("Il link del foglietto non appartiene all'API ufficiale AIFA.")
    request = Request(
        url,
        headers={
            "Accept": "application/pdf",
            "User-Agent": "iFarmaco/1.0",
        },
    )
    try:
        with urlopen(request, timeout=15) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_LEAFLET_SIZE:
                raise ValueError("Il PDF AIFA supera il limite di 25 MB.")
            content = response.read(MAX_LEAFLET_SIZE + 1)
    except (URLError, TimeoutError) as error:
        raise ValueError(f"Impossibile scaricare il foglietto da AIFA ({error}).") from error
    validate_pdf(content)
    return content


def download_missing_leaflets(
    database: Database, store: LeafletStore, limit: int | None = None
) -> tuple[int, list[str]]:
    with database.connect() as connection:
        medicines = connection.execute(
            "SELECT id, bugiardino_url FROM farmaci ORDER BY nome COLLATE NOCASE"
        ).fetchall()
    if limit is not None:
        medicines = medicines[:limit]

    downloaded = 0
    failures = []
    for index, medicine in enumerate(medicines, start=1):
        medicine_id = medicine["id"]
        if store.find(medicine_id):
            continue
        source_url = medicine["bugiardino_url"]
        if not source_url:
            failures.append(f"{medicine_id}: link AIFA assente")
            continue
        try:
            store.save(medicine_id, fetch_aifa_pdf(source_url))
            downloaded += 1
            print(f"[{index}/{len(medicines)}] scaricato {medicine_id}")
        except (OSError, ValueError) as error:
            failures.append(f"{medicine_id}: {error}")
            print(f"[{index}/{len(medicines)}] errore {medicine_id}: {error}")
        time.sleep(0.1)
    return downloaded, failures


if __name__ == "__main__":
    from backend.config import (
        DATABASE_PATH,
        LEAFLETS_DIR,
        MEDICINES_CSV_PATH,
        STOCK_CSV_PATH,
    )
    from backend.database import Database

    parser = argparse.ArgumentParser(description="Scarica i bugiardini AIFA mancanti.")
    parser.add_argument("--limit", type=int, help="Limita il numero di farmaci da provare.")
    arguments = parser.parse_args()
    database = Database(
        DATABASE_PATH,
        medicines_csv=MEDICINES_CSV_PATH,
        stock_csv=STOCK_CSV_PATH,
    )
    database.initialize()
    count, errors = download_missing_leaflets(
        database, LeafletStore(LEAFLETS_DIR), limit=arguments.limit
    )
    print(f"Download completato: {count} PDF nuovi, {len(errors)} non disponibili.")