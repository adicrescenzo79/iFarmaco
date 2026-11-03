import csv
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterator


MEDICINE_FIELDS = (
    "nome",
    "casa_produttrice",
    "formato",
    "quantita",
    "bugiardino_url",
)


class Database:
    def __init__(
        self,
        path: Path,
        medicines_csv: Path | None = None,
        stock_csv: Path | None = None,
    ) -> None:
        self.path = path
        self.medicines_csv = medicines_csv
        self.stock_csv = stock_csv

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS farmaci (
                    id TEXT PRIMARY KEY,
                    nome TEXT NOT NULL,
                    casa_produttrice TEXT NOT NULL DEFAULT '',
                    formato TEXT NOT NULL DEFAULT '',
                    quantita TEXT NOT NULL DEFAULT '',
                    bugiardino_url TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS scorte (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    farmaco_id TEXT NOT NULL REFERENCES farmaci(id)
                        ON UPDATE CASCADE ON DELETE RESTRICT,
                    data_scadenza TEXT NOT NULL,
                    nr_pezzi INTEGER NOT NULL CHECK (nr_pezzi >= 0)
                );

                CREATE INDEX IF NOT EXISTS idx_farmaci_nome
                    ON farmaci(nome COLLATE NOCASE);
                CREATE INDEX IF NOT EXISTS idx_scorte_farmaco_id
                    ON scorte(farmaco_id);
                CREATE INDEX IF NOT EXISTS idx_scorte_data_scadenza
                    ON scorte(data_scadenza);

                CREATE TABLE IF NOT EXISTS app_metadata (
                    chiave TEXT PRIMARY KEY,
                    valore TEXT NOT NULL
                );
                """
            )
            if self.medicines_csv is not None and not self._already_seeded(
                connection, "farmaci_csv_importato"
            ):
                self._import_medicines(connection, self.medicines_csv)
                self._mark_seeded(connection, "farmaci_csv_importato")

            if self.stock_csv is not None and not self._already_seeded(
                connection, "scorte_csv_importato"
            ):
                self._import_stock(connection, self.stock_csv)
                self._mark_seeded(connection, "scorte_csv_importato")

    @staticmethod
    def _already_seeded(connection: sqlite3.Connection, key: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM app_metadata WHERE chiave = ?", (key,)
            ).fetchone()
            is not None
        )

    @staticmethod
    def _mark_seeded(connection: sqlite3.Connection, key: str) -> None:
        connection.execute(
            "INSERT INTO app_metadata (chiave, valore) VALUES (?, ?)", (key, "1")
        )

    @staticmethod
    def _read_csv(path: Path, expected_headers: set[str]) -> list[dict[str, str]]:
        if not path.is_file():
            raise FileNotFoundError(f"File CSV non trovato: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            headers = set(reader.fieldnames or [])
            missing = expected_headers - headers
            if missing:
                raise ValueError(
                    f"Intestazioni mancanti in {path.name}: {', '.join(sorted(missing))}"
                )
            return [
                {key: (value or "").strip() for key, value in row.items() if key}
                for row in reader
            ]

    @classmethod
    def _import_medicines(
        cls, connection: sqlite3.Connection, path: Path
    ) -> None:
        rows = cls._read_csv(path, {"id", *MEDICINE_FIELDS})
        for line_number, row in enumerate(rows, start=2):
            medicine_id = row["id"]
            if not medicine_id or not row["nome"]:
                raise ValueError(
                    f"ID e nome sono obbligatori in {path.name}, riga {line_number}"
                )
            connection.execute(
                """
                INSERT INTO farmaci (
                    id, nome, casa_produttrice, formato, quantita, bugiardino_url
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (medicine_id, *(row[field] for field in MEDICINE_FIELDS)),
            )

    @classmethod
    def _import_stock(cls, connection: sqlite3.Connection, path: Path) -> None:
        rows = cls._read_csv(path, {"farmaco_id", "data_scadenza", "nr_pezzi"})
        for line_number, row in enumerate(rows, start=2):
            try:
                quantity = cls.parse_quantity(row["nr_pezzi"])
                expiry = cls.parse_date(row["data_scadenza"])
            except ValueError as error:
                raise ValueError(
                    f"Dato non valido in {path.name}, riga {line_number}: {error}"
                ) from error
            try:
                connection.execute(
                    """
                    INSERT INTO scorte (farmaco_id, data_scadenza, nr_pezzi)
                    VALUES (?, ?, ?)
                    """,
                    (row["farmaco_id"], expiry, quantity),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    f"Farmaco non presente nel catalogo in {path.name}, "
                    f"riga {line_number}: {row['farmaco_id']}"
                ) from error

    @staticmethod
    def parse_quantity(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("Il numero di pezzi deve essere un intero non negativo")
        try:
            quantity = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Il numero di pezzi deve essere un intero non negativo"
            ) from error
        if quantity < 0 or str(quantity) != str(value).strip():
            raise ValueError("Il numero di pezzi deve essere un intero non negativo")
        return quantity

    @staticmethod
    def parse_date(value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("La data deve essere nel formato AAAA-MM-GG")
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("La data deve essere nel formato AAAA-MM-GG") from error
        if parsed.isoformat() != value:
            raise ValueError("La data deve essere nel formato AAAA-MM-GG")
        return value
