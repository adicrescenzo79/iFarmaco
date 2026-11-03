import logging
import sqlite3
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import cast

from backend.api import RequestHandler
from backend.chat_service import ChatService
from backend.config import (
    DATABASE_PATH,
    DATA_DIR,
    FRONTEND_DIR,
    HOST,
    MEDICINES_CSV_PATH,
    PORT,
    STOCK_CSV_PATH,
)
from backend.database import Database
from backend.leaflets import LeafletStore


def create_server(
    database: Database,
    frontend_dir: Path = FRONTEND_DIR,
    host: str = HOST,
    port: int = PORT,
    leaflet_store: LeafletStore | None = None,
) -> ThreadingHTTPServer:
    handler = type(
        "ConfiguredRequestHandler",
        (RequestHandler,),
        {
            "database": database,
            "frontend_dir": frontend_dir,
            "leaflet_store": leaflet_store or LeafletStore(DATA_DIR / "bugiardini"),
            "chat_service": ChatService(
                database, leaflet_store=leaflet_store or LeafletStore(DATA_DIR / "bugiardini")
            ),
        },
    )
    return ThreadingHTTPServer((host, port), cast(type[RequestHandler], handler))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    database = Database(
        DATABASE_PATH,
        medicines_csv=MEDICINES_CSV_PATH,
        stock_csv=STOCK_CSV_PATH,
    )
    try:
        database.initialize()
    except (OSError, sqlite3.Error, ValueError) as error:
        logging.error("Impossibile inizializzare il database: %s", error)
        raise SystemExit(1) from error

    server = create_server(database)
    logging.info("iFarmaco disponibile su http://%s:%s", HOST, PORT)
    logging.info("Database SQLite: %s", DATABASE_PATH)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Arresto del server")
    finally:
        server.server_close()
