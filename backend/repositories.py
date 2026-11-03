from __future__ import annotations

import sqlite3
from datetime import date
from typing import Any

from backend.database import Database, MEDICINE_FIELDS
from backend.errors import APIError


class MedicineRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def list(self, search: str = "") -> list[dict[str, Any]]:
        pattern = f"%{search.strip()}%"
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT f.*,
                       COUNT(s.id) AS numero_lotti,
                       COALESCE(SUM(s.nr_pezzi), 0) AS pezzi_totali
                FROM farmaci AS f
                LEFT JOIN scorte AS s ON s.farmaco_id = f.id
                WHERE ? = ''
                   OR f.id LIKE ?
                   OR f.nome LIKE ? COLLATE NOCASE
                   OR f.casa_produttrice LIKE ? COLLATE NOCASE
                GROUP BY f.id
                ORDER BY f.nome COLLATE NOCASE, f.id
                """,
                (search.strip(), pattern, pattern, pattern),
            ).fetchall()
            return [dict(row) for row in rows]

    def get(self, medicine_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM farmaci WHERE id = ?", (medicine_id,)
            ).fetchone()
        if row is None:
            raise APIError(404, "Farmaco non trovato.")
        return dict(row)

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        medicine_id = self._required_text(data, "id")
        values = [self._required_text(data, "nome")]
        values.extend(
            self._optional_text(data, field) for field in MEDICINE_FIELDS[1:]
        )
        with self.database.connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO farmaci (
                        id, nome, casa_produttrice, formato, quantita, bugiardino_url
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (medicine_id, *values),
                )
            except sqlite3.IntegrityError as error:
                raise APIError(409, "Esiste già un farmaco con questo ID.") from error
        return self.get(medicine_id)

    def update(self, medicine_id: str, data: dict[str, Any]) -> dict[str, Any]:
        values = [self._required_text(data, "nome")]
        values.extend(
            self._optional_text(data, field) for field in MEDICINE_FIELDS[1:]
        )
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE farmaci
                SET nome = ?, casa_produttrice = ?, formato = ?,
                    quantita = ?, bugiardino_url = ?
                WHERE id = ?
                """,
                (*values, medicine_id),
            )
            if cursor.rowcount == 0:
                raise APIError(404, "Farmaco non trovato.")
        return self.get(medicine_id)

    @staticmethod
    def _required_text(data: dict[str, Any], field: str) -> str:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            raise APIError(400, f"Il campo '{field}' è obbligatorio.")
        return value.strip()

    @staticmethod
    def _optional_text(data: dict[str, Any], field: str) -> str:
        value = data.get(field, "")
        if not isinstance(value, str):
            raise APIError(400, f"Il campo '{field}' deve essere testo.")
        return value.strip()


class StockRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def list(self, search: str = "") -> list[dict[str, Any]]:
        pattern = f"%{search.strip()}%"
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT s.id, s.farmaco_id, f.nome AS farmaco_nome,
                       s.data_scadenza, s.nr_pezzi
                FROM scorte AS s
                JOIN farmaci AS f ON f.id = s.farmaco_id
                WHERE ? = ''
                   OR s.farmaco_id LIKE ?
                   OR f.nome LIKE ? COLLATE NOCASE
                   OR s.data_scadenza LIKE ?
                ORDER BY s.data_scadenza, f.nome COLLATE NOCASE, s.id
                """,
                (search.strip(), pattern, pattern, pattern),
            ).fetchall()
            return [dict(row) for row in rows]

    def for_medicine(self, medicine_id: str) -> list[dict[str, Any]]:
        today = date.today().isoformat()
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, data_scadenza, nr_pezzi,
                       CASE WHEN data_scadenza >= ? THEN 1 ELSE 0 END AS non_scaduto
                FROM scorte
                WHERE farmaco_id = ?
                ORDER BY data_scadenza, id
                """,
                (today, medicine_id),
            ).fetchall()
            return [dict(row) for row in rows]

    def get(self, stock_id: int) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT s.id, s.farmaco_id, f.nome AS farmaco_nome,
                       s.data_scadenza, s.nr_pezzi
                FROM scorte AS s
                JOIN farmaci AS f ON f.id = s.farmaco_id
                WHERE s.id = ?
                """,
                (stock_id,),
            ).fetchone()
        if row is None:
            raise APIError(404, "Lotto di scorta non trovato.")
        return dict(row)

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        medicine_id = self._required_text(data, "farmaco_id")
        expiry = self._parse_date(data.get("data_scadenza"))
        quantity = self._parse_quantity(data.get("nr_pezzi"))
        with self.database.connect() as connection:
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO scorte (farmaco_id, data_scadenza, nr_pezzi)
                    VALUES (?, ?, ?)
                    """,
                    (medicine_id, expiry, quantity),
                )
            except sqlite3.IntegrityError as error:
                raise APIError(400, "Il farmaco selezionato non esiste.") from error
            stock_id = cursor.lastrowid
        return self.get(stock_id)

    def update(self, stock_id: int, data: dict[str, Any]) -> dict[str, Any]:
        medicine_id = self._required_text(data, "farmaco_id")
        expiry = self._parse_date(data.get("data_scadenza"))
        quantity = self._parse_quantity(data.get("nr_pezzi"))
        with self.database.connect() as connection:
            try:
                cursor = connection.execute(
                    """
                    UPDATE scorte
                    SET farmaco_id = ?, data_scadenza = ?, nr_pezzi = ?
                    WHERE id = ?
                    """,
                    (medicine_id, expiry, quantity, stock_id),
                )
            except sqlite3.IntegrityError as error:
                raise APIError(400, "Il farmaco selezionato non esiste.") from error
            if cursor.rowcount == 0:
                raise APIError(404, "Lotto di scorta non trovato.")
        return self.get(stock_id)

    def delete(self, stock_id: int) -> None:
        with self.database.connect() as connection:
            cursor = connection.execute("DELETE FROM scorte WHERE id = ?", (stock_id,))
            if cursor.rowcount == 0:
                raise APIError(404, "Lotto di scorta non trovato.")

    def _parse_date(self, value: Any) -> str:
        try:
            return self.database.parse_date(value)
        except ValueError as error:
            raise APIError(400, str(error) + ".") from error

    def _parse_quantity(self, value: Any) -> int:
        try:
            return self.database.parse_quantity(value)
        except ValueError as error:
            raise APIError(400, str(error) + ".") from error

    @staticmethod
    def _required_text(data: dict[str, Any], field: str) -> str:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            raise APIError(400, f"Il campo '{field}' è obbligatorio.")
        return value.strip()
