import json
import logging
import re
import sqlite3
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from backend.config import MAX_REQUEST_SIZE, MAX_UPLOAD_SIZE
from backend.chat_service import ChatService, ChatUnavailableError
from backend.database import Database
from backend.errors import APIError
from backend.leaflets import LeafletStore, fetch_aifa_pdf
from backend.repositories import MedicineRepository, StockRepository


class RequestHandler(BaseHTTPRequestHandler):
    database: Database
    frontend_dir: Path
    leaflet_store: LeafletStore
    chat_service: ChatService

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if not parsed.path.startswith("/api/"):
            self._serve_frontend(parsed.path)
            return
        try:
            query = parse_qs(parsed.query)
            if parsed.path == "/api/medicines":
                result = MedicineRepository(self.database).list(
                    query.get("q", [""])[0]
                )
            elif re.fullmatch(r"/api/medicines/[^/]+/leaflet", parsed.path):
                self._serve_leaflet(unquote(parsed.path.split("/")[3]))
                return
            elif parsed.path == "/api/chat/config":
                result = self.chat_service.configuration()
            elif re.fullmatch(r"/api/chat/session/[^/]+", parsed.path):
                session_id = unquote(parsed.path.rsplit("/", 1)[1])
                result = self.chat_service.session(session_id)
            elif parsed.path == "/api/stock":
                result = StockRepository(self.database).list(query.get("q", [""])[0])
            elif parsed.path == "/api/summary":
                result = self._get_summary()
            else:
                raise APIError(404, "Endpoint non trovato.")
            self._send_json(200, result)
        except APIError as error:
            self._send_error_json(error.status, error.message)
        except ValueError as error:
            self._send_error_json(400, str(error))
        except sqlite3.Error:
            logging.exception("Errore di lettura dal database")
            self._send_error_json(500, "Errore durante la lettura del database.")

    def do_POST(self) -> None:
        try:
            if urlsplit(self.path).path == "/api/medicines":
                data, leaflet = self._read_multipart_body()
                medicine_id = data.get("id", "")
                try:
                    staged_path = self.leaflet_store.stage(medicine_id, leaflet)
                except ValueError as error:
                    raise APIError(400, str(error)) from error
                try:
                    result = MedicineRepository(self.database).create(data)
                    try:
                        self.leaflet_store.commit(medicine_id, staged_path)
                    except OSError as error:
                        with self.database.connect() as connection:
                            connection.execute(
                                "DELETE FROM farmaci WHERE id = ?", (medicine_id,)
                            )
                        raise APIError(
                            500, "Impossibile salvare il bugiardino caricato."
                        ) from error
                finally:
                    staged_path.unlink(missing_ok=True)
            elif urlsplit(self.path).path == "/api/chat":
                result = self.chat_service.respond(self._read_json_body())
            elif urlsplit(self.path).path == "/api/stock":
                data = self._read_json_body()
                result = StockRepository(self.database).create(data)
            else:
                raise APIError(404, "Endpoint non trovato.")
            status = 200 if urlsplit(self.path).path == "/api/chat" else 201
            self._send_json(status, result)
        except APIError as error:
            self._send_error_json(error.status, error.message)
        except ChatUnavailableError as error:
            self._send_error_json(503, str(error))
        except ValueError as error:
            self._send_error_json(400, str(error))
        except sqlite3.Error:
            logging.exception("Errore di scrittura nel database")
            self._send_error_json(500, "Errore durante il salvataggio.")
        except Exception:
            logging.exception("Errore durante la risposta RAG")
            self._send_error_json(
                502, "La chat non è riuscita. Controlla configurazione e connessione."
            )

    def do_PUT(self) -> None:
        parsed = urlsplit(self.path)
        parts = parsed.path.strip("/").split("/")
        try:
            data = self._read_json_body()
            if len(parts) == 3 and parts[:2] == ["api", "medicines"]:
                result = MedicineRepository(self.database).update(
                    unquote(parts[2]), data
                )
            elif len(parts) == 3 and parts[:2] == ["api", "stock"]:
                result = StockRepository(self.database).update(
                    self._parse_id(parts[2]), data
                )
            else:
                raise APIError(404, "Endpoint non trovato.")
            self._send_json(200, result)
        except APIError as error:
            self._send_error_json(error.status, error.message)
        except sqlite3.Error:
            logging.exception("Errore di scrittura nel database")
            self._send_error_json(500, "Errore durante il salvataggio.")

    def do_DELETE(self) -> None:
        parts = urlsplit(self.path).path.strip("/").split("/")
        try:
            if len(parts) != 3 or parts[:2] != ["api", "stock"]:
                raise APIError(404, "Endpoint non trovato.")
            StockRepository(self.database).delete(self._parse_id(parts[2]))
            self._send_json(200, {"ok": True})
        except APIError as error:
            self._send_error_json(error.status, error.message)
        except sqlite3.Error:
            logging.exception("Errore di scrittura nel database")
            self._send_error_json(500, "Errore durante l'eliminazione.")

    def _get_summary(self) -> dict[str, int]:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT (SELECT COUNT(*) FROM farmaci) AS farmaci,
                       (SELECT COUNT(*) FROM scorte) AS lotti,
                       (SELECT COALESCE(SUM(nr_pezzi), 0) FROM scorte) AS pezzi
                """
            ).fetchone()
        return dict(row)

    @staticmethod
    def _parse_id(value: str) -> int:
        if not re.fullmatch(r"\d+", value):
            raise APIError(400, "ID non valido.")
        return int(value)

    def _read_json_body(self) -> dict[str, Any]:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise APIError(400, "Lunghezza della richiesta non valida.") from error
        if size <= 0 or size > MAX_REQUEST_SIZE:
            raise APIError(400, "La richiesta è vuota o troppo grande.")
        try:
            body = json.loads(self.rfile.read(size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise APIError(
                400, "Il corpo della richiesta deve essere JSON valido."
            ) from error
        if not isinstance(body, dict):
            raise APIError(400, "Il corpo della richiesta deve essere un oggetto JSON.")
        return body

    def _read_multipart_body(self) -> tuple[dict[str, str], bytes]:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data;"):
            raise APIError(400, "Per aggiungere un farmaco carica il bugiardino PDF.")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise APIError(400, "Lunghezza della richiesta non valida.") from error
        if size <= 0 or size > MAX_UPLOAD_SIZE:
            raise APIError(400, "La richiesta è vuota o supera il limite consentito.")
        raw_body = self.rfile.read(size)
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
            + raw_body
        )
        if not message.is_multipart():
            raise APIError(400, "Formato del caricamento non valido.")

        fields: dict[str, str] = {}
        leaflet: bytes | None = None
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            payload = part.get_payload(decode=True) or b""
            if part.get_filename():
                if name != "bugiardino" or leaflet is not None:
                    raise APIError(400, "Carica un solo bugiardino PDF.")
                if not part.get_filename().lower().endswith(".pdf"):
                    raise APIError(400, "Il bugiardino deve essere un file PDF.")
                leaflet = payload
            elif name:
                charset = part.get_content_charset() or "utf-8"
                try:
                    fields[name] = payload.decode(charset).strip()
                except (LookupError, UnicodeDecodeError) as error:
                    raise APIError(400, "Testo del modulo non valido.") from error
        if leaflet is None:
            raise APIError(400, "Il caricamento del bugiardino PDF è obbligatorio.")
        return fields, leaflet

    def _serve_leaflet(self, medicine_id: str) -> None:
        try:
            medicine = MedicineRepository(self.database).get(medicine_id)
            path = self.leaflet_store.find(medicine_id)
            if path is None and medicine["bugiardino_url"]:
                content = fetch_aifa_pdf(medicine["bugiardino_url"])
                path = self.leaflet_store.save(medicine_id, content)
            if path is None:
                raise APIError(404, "Bugiardino non disponibile.")
            content = path.read_bytes()
        except APIError as error:
            self._send_error_json(error.status, error.message)
            return
        except ValueError as error:
            self._send_error_json(502, str(error))
            return
        except OSError:
            self._send_error_json(500, "Impossibile leggere il bugiardino.")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Content-Disposition", "inline; filename=bugiardino.pdf")
        self.send_header("Cache-Control", "private, max-age=3600")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _serve_frontend(self, path: str) -> None:
        requested = "index.html" if path in ("", "/") else path.lstrip("/")
        if requested not in {"index.html", "styles.css", "app.js"}:
            self._send_error_json(404, "Pagina non trovata.")
            return
        file_path = self.frontend_dir / requested
        try:
            content = file_path.read_bytes()
        except FileNotFoundError:
            self._send_error_json(404, "File dell'interfaccia non trovato.")
            return
        content_type = {
            "index.html": "text/html; charset=utf-8",
            "styles.css": "text/css; charset=utf-8",
            "app.js": "text/javascript; charset=utf-8",
        }[requested]
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: int, payload: Any) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def log_message(self, format_string: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), format_string % args)
