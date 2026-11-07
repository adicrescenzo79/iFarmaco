import csv
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from time import monotonic
from typing import Any, Iterator


MAX_READ_QUERY_LENGTH = 4_000
MAX_READ_QUERY_ROWS = 100
MAX_READ_QUERY_RESULT_CHARS = 100_000
READ_QUERY_TIMEOUT_SECONDS = 0.5
READ_QUERY_TABLES = frozenset({"farmaci", "scorte"})
READ_QUERY_BLOCKED_FUNCTIONS = frozenset(
    {"load_extension", "randomblob", "readfile", "writefile", "zeroblob"}
)
READ_QUERY_PATTERN = re.compile(r"(?:SELECT|WITH)\b", re.IGNORECASE)


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

    def query_readonly(self, query: str) -> dict[str, Any]:
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > MAX_READ_QUERY_LENGTH
            or not READ_QUERY_PATTERN.match(query.lstrip())
        ):
            raise ValueError(
                "Query non valida: inserisci una SELECT di al massimo "
                f"{MAX_READ_QUERY_LENGTH} caratteri."
            )

        connection = sqlite3.connect(
            f"{self.path.resolve().as_uri()}?mode=ro", uri=True
        )
        connection.row_factory = sqlite3.Row
        deadline = monotonic() + READ_QUERY_TIMEOUT_SECONDS
        timed_out = False

        def authorize(
            action: int,
            first_argument: str | None,
            second_argument: str | None,
            database_name: str | None,
            trigger_name: str | None,
        ) -> int:
            del database_name, trigger_name
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ:
                return (
                    sqlite3.SQLITE_OK
                    if (first_argument or "").casefold() in READ_QUERY_TABLES
                    else sqlite3.SQLITE_DENY
                )
            if action == sqlite3.SQLITE_FUNCTION:
                function_name = (second_argument or first_argument or "").casefold()
                return (
                    sqlite3.SQLITE_DENY
                    if function_name in READ_QUERY_BLOCKED_FUNCTIONS
                    else sqlite3.SQLITE_OK
                )
            if action == getattr(sqlite3, "SQLITE_RECURSIVE", -1):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        def interrupt_long_query() -> int:
            nonlocal timed_out
            timed_out = monotonic() >= deadline
            return int(timed_out)

        try:
            set_limit = getattr(connection, "setlimit", None)
            if set_limit is not None:
                set_limit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
                set_limit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, MAX_READ_QUERY_LENGTH)
                set_limit(sqlite3.SQLITE_LIMIT_COLUMN, 50)
            connection.set_authorizer(authorize)
            connection.set_progress_handler(interrupt_long_query, 1_000)
            cursor = connection.execute(query)
            rows = []
            result_size = 0
            truncated = False
            for row in cursor:
                if len(rows) >= MAX_READ_QUERY_ROWS:
                    truncated = True
                    break
                values = {
                    key: (
                        value.decode("utf-8", errors="replace")
                        if isinstance(value, bytes)
                        else value
                    )
                    for key, value in dict(row).items()
                }
                row_size = len(json.dumps(values, ensure_ascii=False))
                if result_size + row_size > MAX_READ_QUERY_RESULT_CHARS:
                    truncated = True
                    break
                rows.append(values)
                result_size += row_size
            return {
                "columns": [column[0] for column in cursor.description or ()],
                "rows": [dict(row) for row in rows],
                "truncated": truncated,
            }
        except sqlite3.Error as error:
            if timed_out:
                raise ValueError(
                    "La query è stata interrotta perché ha superato il limite di tempo."
                ) from error
            raise ValueError(
                "Query non valida o non consentita. Sono ammesse solo letture "
                "dalle tabelle farmaci e scorte."
            ) from error
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
