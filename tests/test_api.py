import json
import os
import tempfile
import threading
import unittest
from io import BytesIO
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch
from pypdf import PdfWriter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.database import Database
from backend.leaflets import LeafletStore
from backend.server import create_server


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        database = Database(Path(cls.temp_dir.name) / "api.sqlite3")
        database.initialize()
        cls.leaflet_store = LeafletStore(Path(cls.temp_dir.name) / "bugiardini")
        cls.server = create_server(database, port=0, leaflet_store=cls.leaflet_store)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.temp_dir.cleanup()

    def request(
        self, method: str, path: str, payload: dict[str, object] | None = None
    ) -> tuple[int, object]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())
        except URLError as error:
            self.fail(f"API non raggiungibile: {error}")

    def request_multipart(
        self,
        path: str,
        fields: dict[str, str],
        filename: str,
        content: bytes,
    ) -> tuple[int, object]:
        boundary = "ifarmaco-test-boundary"
        body = bytearray()
        for name, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
            )
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            (
                'Content-Disposition: form-data; name="bugiardino"; '
                f'filename="{filename}"\r\n'
                "Content-Type: application/pdf\r\n\r\n"
            ).encode()
        )
        body.extend(content)
        body.extend(f"\r\n--{boundary}--\r\n".encode())
        request = Request(
            self.base_url + path,
            data=bytes(body),
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())
        except URLError as error:
            self.fail(f"API non raggiungibile: {error}")

    @staticmethod
    def sample_pdf() -> bytes:
        output = BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(output)
        return output.getvalue()

    def test_medicine_and_stock_crud(self) -> None:
        status, medicine = self.request_multipart(
            "/api/medicines",
            {
                "id": "00099",
                "nome": "Farmaco API",
                "casa_produttrice": "Produttore",
                "formato": "Compressa",
                "quantita": "10 mg",
                "bugiardino_url": "",
            },
            "bugiardino.pdf",
            self.sample_pdf(),
        )
        self.assertEqual(status, 201)
        self.assertIsInstance(medicine, dict)
        self.assertTrue(self.leaflet_store.find("00099").is_file())
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request("GET", "/api/medicines/00099/leaflet")
        response = connection.getresponse()
        leaflet = response.read()
        connection.close()
        self.assertEqual(response.status, 200)
        self.assertTrue(leaflet.startswith(b"%PDF-"))

        status, updated_medicine = self.request(
            "PUT",
            "/api/medicines/00099",
            {
                "nome": "Farmaco API modificato",
                "casa_produttrice": "Produttore",
                "formato": "Compressa",
                "quantita": "20 mg",
                "bugiardino_url": "",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(updated_medicine["nome"], "Farmaco API modificato")

        status, lots = self.request(
            "POST",
            "/api/stock",
            {"farmaco_id": "00099", "data_scadenza": "2027-01-31", "nr_pezzi": 4},
        )
        self.assertEqual(status, 201)
        self.assertIsInstance(lots, dict)
        stock_id = lots["id"]

        status, updated = self.request(
            "PUT",
            f"/api/stock/{stock_id}",
            {"farmaco_id": "00099", "data_scadenza": "2028-01-31", "nr_pezzi": 9},
        )
        self.assertEqual(status, 200)
        self.assertEqual(updated["nr_pezzi"], 9)

        status, summary = self.request("GET", "/api/summary")
        self.assertEqual(status, 200)
        self.assertEqual(summary["farmaci"], 1)

        status, _ = self.request("DELETE", f"/api/stock/{stock_id}")
        self.assertEqual(status, 200)

    def test_new_medicine_requires_a_leaflet_upload(self) -> None:
        status, payload = self.request(
            "POST",
            "/api/medicines",
            {"id": "00098", "nome": "Senza bugiardino"},
        )
        self.assertEqual(status, 400)
        self.assertIn("bugiardino", payload["error"])

        status, payload = self.request_multipart(
            "/api/medicines",
            {"id": "00097", "nome": "PDF non valido"},
            "bugiardino.pdf",
            b"%PDF-1.4 not-a-pdf",
        )
        self.assertEqual(status, 400)
        self.assertIn("PDF", payload["error"])

    def test_chat_reports_missing_api_key_without_calling_provider(self) -> None:
        status, config = self.request("GET", "/api/chat/config")
        self.assertEqual(status, 200)
        self.assertIn("models", config)
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            status, payload = self.request(
                "POST",
                "/api/chat",
                {
                    "session_id": "setup-check",
                    "medicine_id": "00099",
                    "message": "Domanda di prova",
                },
            )
        self.assertEqual(status, 503)
        self.assertIn("OPENAI_API_KEY", payload["error"])

    def test_frontend_is_served_by_backend(self) -> None:
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request("GET", "/")
        response = connection.getresponse()
        page = response.read().decode("utf-8")
        connection.close()
        self.assertEqual(response.status, 200)
        self.assertIn("iFarmaco", page)


if __name__ == "__main__":
    unittest.main()
