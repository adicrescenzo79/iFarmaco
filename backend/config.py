import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
FRONTEND_DIR = ROOT_DIR / "frontend"
DATABASE_PATH = DATA_DIR / "ifarmaco.sqlite3"
MEDICINES_CSV_PATH = DATA_DIR / "farmaci.csv"
STOCK_CSV_PATH = DATA_DIR / "scorte_farmaci.csv"
HOST = "127.0.0.1"
PORT = 8000
PORT = int(os.getenv("IFARMACO_PORT", str(PORT)))
MAX_REQUEST_SIZE = 1_000_000
MAX_UPLOAD_SIZE = 26 * 1024 * 1024
LEAFLETS_DIR = DATA_DIR / "bugiardini"
CHAT_SESSIONS_DIR = DATA_DIR / "chat_sessions"
CHROMA_DIR = DATA_DIR / "chroma"
