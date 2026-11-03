import tempfile
import unittest
from pathlib import Path

from backend.database import Database
from backend.errors import APIError
from backend.repositories import MedicineRepository, StockRepository


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temp_dir.name) / "test.sqlite3")
        self.database.initialize()
        self.medicines = MedicineRepository(self.database)
        self.stock = StockRepository(self.database)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_create_and_update_medicine(self) -> None:
        medicine = self.medicines.create(
            {
                "id": "00001",
                "nome": "Farmaco di prova",
                "casa_produttrice": "Produttore",
            }
        )
        self.assertEqual(medicine["id"], "00001")
        updated = self.medicines.update(
            "00001",
            {
                "nome": "Farmaco aggiornato",
                "casa_produttrice": "Produttore",
                "formato": "Compressa",
                "quantita": "20 mg",
                "bugiardino_url": "",
            },
        )
        self.assertEqual(updated["nome"], "Farmaco aggiornato")

    def test_stock_requires_existing_medicine_and_valid_values(self) -> None:
        with self.assertRaisesRegex(APIError, "non esiste"):
            self.stock.create(
                {
                    "farmaco_id": "missing",
                    "data_scadenza": "2027-01-31",
                    "nr_pezzi": 3,
                }
            )
        with self.assertRaisesRegex(APIError, "data"):
            self.stock.create(
                {
                    "farmaco_id": "missing",
                    "data_scadenza": "31-01-2027",
                    "nr_pezzi": 3,
                }
            )
        with self.assertRaisesRegex(APIError, "pezzi"):
            self.stock.create(
                {
                    "farmaco_id": "missing",
                    "data_scadenza": "2027-01-31",
                    "nr_pezzi": -1,
                }
            )

    def test_initialize_imports_csvs_once(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Database(
                Path(temp_dir) / "seed.sqlite3",
                medicines_csv=root / "data" / "farmaci.csv",
                stock_csv=root / "data" / "scorte_farmaci.csv",
            )
            database.initialize()
            initial_summary = self._summary(database)
            database.initialize()
            self.assertEqual(self._summary(database), initial_summary)
            self.assertEqual(initial_summary["farmaci"], 100)
            self.assertGreaterEqual(initial_summary["lotti"], 120)

    @staticmethod
    def _summary(database: Database) -> dict[str, int]:
        with database.connect() as connection:
            row = connection.execute(
                "SELECT (SELECT COUNT(*) FROM farmaci) AS farmaci, "
                "(SELECT COUNT(*) FROM scorte) AS lotti"
            ).fetchone()
        return dict(row)


if __name__ == "__main__":
    unittest.main()
