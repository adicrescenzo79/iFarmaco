import hashlib
import json
import os
import re
import tempfile
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from backend.config import CHROMA_DIR, CHAT_SESSIONS_DIR, LEAFLETS_DIR, ROOT_DIR
from backend.database import Database
from backend.leaflets import LeafletStore, fetch_aifa_pdf
from backend.repositories import MedicineRepository, StockRepository

load_dotenv(ROOT_DIR / ".env")

CHAT_MODELS = ("gpt-4o-mini", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
MAX_MESSAGE_LENGTH = 4_000
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
INSTRUCTIONS = """Sei l'assistente informativo di iFarmaco. Per le domande cliniche
usa solo il contenuto dei foglietti illustrativi forniti nel turno corrente e
cita la fonte nel formato [Nome farmaco, p. N]. Se i documenti non contengono la
risposta, dichiaralo chiaramente e non inventare informazioni. Non formulare
diagnosi né prescrivere, sospendere o modificare terapie; per consigli
personali rimanda a medico o farmacista. In caso di possibile emergenza, invita
l'utente a contattare subito i servizi di emergenza. Per domande su quantità,
disponibilità o scorte correnti usa get_medicine_stock; non dedurre le quantità
da memoria o dai bugiardini. Distingui pezzi scaduti e non scaduti e non
sommare prodotti distinti senza specificarlo."""
INVENTORY_TOOL = {
    "type": "function",
    "name": "get_medicine_stock",
    "description": (
        "Legge dal database SQLite i lotti e le quantità correnti dei farmaci. "
        "Usa questo tool quando l'utente chiede quante confezioni o pezzi ha, "
        "la disponibilità o le scorte di uno o più farmaci. Non modifica dati."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "medicine_name": {
                "type": "string",
                "description": "Nome o principio attivo del farmaco da cercare.",
            }
        },
        "required": ["medicine_name"],
        "additionalProperties": False,
    },
    "strict": True,
}
INVENTORY_TERMS = re.compile(
    r"\b(scorte?|giacenze?|disponibilit[aà])\b", re.IGNORECASE
)
INVENTORY_QUANTITY = re.compile(
    r"\b(quante?|quanti)\b.*\b(ho|restano|rimangono|disponibili|"
    r"a disposizione)\b",
    re.IGNORECASE,
)
DOSAGE_FREQUENCY = re.compile(r"\b(quante?|quanti)\s+volte\b", re.IGNORECASE)


class ChatUnavailableError(Exception):
    pass


class ChatService:
    def __init__(
        self,
        database: Database,
        leaflet_store: LeafletStore | None = None,
        sessions_dir: Path = CHAT_SESSIONS_DIR,
        chroma_dir: Path = CHROMA_DIR,
    ) -> None:
        self.database = database
        self.leaflet_store = leaflet_store or LeafletStore(LEAFLETS_DIR)
        self.sessions_dir = sessions_dir
        self.chroma_dir = chroma_dir
        self._client: Any = None
        self._collection: Any = None

    @staticmethod
    def configuration() -> dict[str, Any]:
        default_model = os.getenv("OPENAI_CHAT_MODEL", CHAT_MODELS[0])
        if default_model not in CHAT_MODELS:
            default_model = CHAT_MODELS[0]
        return {
            "configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
            "models": list(CHAT_MODELS),
            "default_model": default_model,
        }

    def session(self, session_id: str) -> dict[str, Any]:
        path = self._session_path(session_id)
        if not path.is_file():
            return {"messages": [], "medicine_id": "", "medicine_ids": []}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("La sessione salvata non è leggibile.") from error
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError("La sessione salvata non è valida.")
        return {
            "messages": [
                message
                for message in messages
                if isinstance(message, dict)
                and message.get("role") in {"user", "assistant"}
                and isinstance(message.get("content"), str)
            ],
            "medicine_id": str(payload.get("medicine_id", "")),
            "medicine_ids": [
                str(medicine_id)
                for medicine_id in payload.get(
                    "medicine_ids", [payload.get("medicine_id", "")]
                )
                if medicine_id
            ],
        }

    def respond(self, payload: dict[str, Any]) -> dict[str, Any]:
        configuration = self.configuration()
        if not configuration["configured"]:
            raise ChatUnavailableError("Configura OPENAI_API_KEY nel file .env.")

        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Scrivi una domanda prima di inviarla.")
        if len(message) > MAX_MESSAGE_LENGTH:
            raise ValueError("La domanda supera il limite di 4.000 caratteri.")
        session_id = payload.get("session_id", "")
        session_path = self._session_path(session_id)
        saved = self.session(session_id)

        selected_id = payload.get("medicine_id")
        mentioned = self._find_mentioned_medicines(message)
        if mentioned:
            medicines = mentioned
        elif isinstance(selected_id, str) and selected_id:
            medicines = [MedicineRepository(self.database).get(selected_id)]
        else:
            medicines = [
                MedicineRepository(self.database).get(medicine_id)
                for medicine_id in saved["medicine_ids"]
            ]
        model = payload.get("model", configuration["default_model"])
        if model not in CHAT_MODELS:
            raise ValueError("Modello non supportato.")
        try:
            top_k = int(payload.get("top_k", 4))
            budget = int(payload.get("budget_tokens", 700))
        except (TypeError, ValueError) as error:
            raise ValueError("Configurazione RAG non valida.") from error
        if not 1 <= top_k <= 8 or not 150 <= budget <= 4_000:
            raise ValueError("Fonti o budget memoria fuori dai limiti consentiti.")
        memory_mode = payload.get("memory_mode", "buffer")
        if memory_mode not in {"buffer", "summary"}:
            raise ValueError("Modalità memoria non supportata.")

        history = saved["messages"][-100:]
        inventory_question = self._is_inventory_question(message)
        if not medicines and not inventory_question:
            answer = (
                "A quale farmaco ti riferisci? Puoi scriverne il nome nella domanda; "
                "per confrontarne più di uno, includi tutti i nomi."
            )
            history.extend(
                [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": answer, "sources": []},
                ]
            )
            self._truncate(history, budget)
            self._save_session(session_path, history, [])
            return {
                "session_id": session_id,
                "medicine_id": "",
                "medicine_ids": [],
                "medicine_names": [],
                "answer": answer,
                "searched_query": message,
                "sources": [],
                "messages": history,
                "needs_clarification": True,
            }
        if len(medicines) > 4:
            raise ValueError("Per volta puoi confrontare al massimo quattro farmaci.")

        client = self._openai_client()
        if inventory_question:
            answer, inventory_results = self._answer_inventory(
                client,
                model,
                history,
                message,
                [medicine["nome"] for medicine in medicines],
            )
            medicine_ids = list(
                dict.fromkeys(
                    match["medicine_id"]
                    for result in inventory_results
                    for match in result.get("matches", [])
                )
            )
            medicine_names = [
                MedicineRepository(self.database).get(medicine_id)["nome"]
                for medicine_id in medicine_ids
            ]
            searched_query = message
            chunks = []
        else:
            medicine_names = [medicine["nome"] for medicine in medicines]
            searched_query = self._condense(
                client, model, history, message, ", ".join(medicine_names)
            )
            chunks = []
            for medicine in medicines:
                chunks.extend(self._retrieve(client, medicine, searched_query, top_k))
            chunks.sort(key=lambda chunk: chunk["score"], reverse=True)
            chunks = chunks[:top_k]
            answer = self._generate(client, model, chunks, searched_query, history)
            medicine_ids = [medicine["id"] for medicine in medicines]
        sources = [
            {
                "id": chunk["id"],
                "medicine_id": chunk["medicine_id"],
                "page": chunk["page"],
                "score": round(chunk["score"], 3),
                "url": f"/api/medicines/{chunk['medicine_id']}/leaflet#page={chunk['page']}",
            }
            for chunk in chunks
        ]
        history.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer, "sources": sources},
            ]
        )
        if memory_mode == "summary":
            self._summarize_if_needed(client, model, history, budget)
        else:
            self._truncate(history, budget)
        self._save_session(session_path, history, medicine_ids)

        return {
            "session_id": session_id,
            "medicine_id": medicine_ids[0] if medicine_ids else "",
            "medicine_ids": medicine_ids,
            "medicine_names": medicine_names,
            "answer": answer,
            "searched_query": searched_query,
            "sources": sources,
            "messages": history,
        }

    def _answer_inventory(
        self,
        client: Any,
        model: str,
        history: list[dict[str, str]],
        message: str,
        selected_medicines: list[str],
    ) -> tuple[str, list[dict[str, Any]]]:
        medicine_context = (
            ", ".join(selected_medicines) if selected_medicines else "nessuno selezionato"
        )
        inputs = [
            {"role": item["role"], "content": item["content"]}
            for item in history
        ]
        inputs.append(
            {
                "role": "user",
                "content": (
                    f"Farmaci selezionati nel filtro: {medicine_context}.\n"
                    f"Domanda sulle scorte: {message}"
                ),
            }
        )
        results: list[dict[str, Any]] = []

        for _ in range(3):
            response = client.responses.create(
                model=model,
                instructions=INSTRUCTIONS,
                input=inputs,
                tools=[INVENTORY_TOOL],
                tool_choice="auto",
            )
            calls = [
                item
                for item in response.output
                if getattr(item, "type", "") == "function_call"
            ]
            if not calls:
                return response.output_text.strip(), results

            for call in calls:
                if call.name != "get_medicine_stock":
                    output = {"error": "Tool non supportato."}
                else:
                    try:
                        arguments = json.loads(call.arguments)
                        output = self._execute_inventory_tool(arguments)
                    except (TypeError, ValueError, json.JSONDecodeError) as error:
                        output = {"error": f"Argomenti del tool non validi: {error}"}
                results.append(output)
                if hasattr(call, "model_dump"):
                    inputs.append(call.model_dump(exclude_none=True))
                else:
                    inputs.append(
                        {
                            "type": "function_call",
                            "name": call.name,
                            "arguments": call.arguments,
                            "call_id": call.call_id,
                        }
                    )
                inputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(output, ensure_ascii=False),
                    }
                )

        response = client.responses.create(
            model=model,
            instructions=INSTRUCTIONS,
            input=inputs,
            tools=[INVENTORY_TOOL],
            tool_choice="none",
        )
        return response.output_text.strip(), results

    def _execute_inventory_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        medicine_name = arguments.get("medicine_name")
        if not isinstance(medicine_name, str) or not medicine_name.strip():
            return {"error": "medicine_name è obbligatorio."}
        medicine_name = medicine_name.strip()[:120]
        medicines = self._match_inventory_medicines(medicine_name)
        if not medicines:
            return {"query": medicine_name, "matches": []}

        stock = StockRepository(self.database)
        matches = []
        for medicine in medicines[:8]:
            lots = stock.for_medicine(medicine["id"])
            matches.append(
                {
                    "medicine_id": medicine["id"],
                    "name": medicine["nome"],
                    "numero_lotti": len(lots),
                    "pezzi_totali": sum(lot["nr_pezzi"] for lot in lots),
                    "pezzi_non_scaduti": sum(
                        lot["nr_pezzi"] for lot in lots if lot["non_scaduto"]
                    ),
                    "pezzi_scaduti": sum(
                        lot["nr_pezzi"] for lot in lots if not lot["non_scaduto"]
                    ),
                    "lotti": [
                        {
                            "data_scadenza": lot["data_scadenza"],
                            "nr_pezzi": lot["nr_pezzi"],
                            "non_scaduto": bool(lot["non_scaduto"]),
                        }
                        for lot in lots
                    ],
                }
            )
        return {"query": medicine_name, "matches": matches}

    def _match_inventory_medicines(self, query: str) -> list[dict[str, Any]]:
        normalized_query = self._normalize_medicine_name(query)
        medicines = MedicineRepository(self.database).list()
        exact_matches = [
            medicine
            for medicine in medicines
            if normalized_query in self._normalize_medicine_name(medicine["nome"])
            or self._normalize_medicine_name(medicine["nome"]) in normalized_query
        ]
        if exact_matches:
            return exact_matches

        query_tokens = normalized_query.split()
        scored = []
        for medicine in medicines:
            name_tokens = self._normalize_medicine_name(medicine["nome"]).split()
            score = max(
                (
                    SequenceMatcher(None, query_token, name_token).ratio()
                    for query_token in query_tokens
                    for name_token in name_tokens
                ),
                default=0,
            )
            scored.append((score, medicine))
        best_score = max((score for score, _ in scored), default=0)
        if best_score < 0.72:
            return []
        return [
            medicine
            for score, medicine in sorted(scored, key=lambda item: item[0], reverse=True)
            if score >= max(0.72, best_score - 0.03)
        ]

    @staticmethod
    def _is_inventory_question(message: str) -> bool:
        if INVENTORY_TERMS.search(message):
            return True
        return bool(
            INVENTORY_QUANTITY.search(message)
            and not DOSAGE_FREQUENCY.search(message)
        )

    def _openai_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def _vector_collection(self) -> Any:
        if self._collection is None:
            import chromadb

            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self._collection = client.get_or_create_collection(
                "ifarmaco_leaflets", metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def _retrieve(
        self, client: Any, medicine: dict[str, Any], query: str, top_k: int
    ) -> list[dict[str, Any]]:
        path = self.leaflet_store.find(medicine["id"])
        if path is None:
            source_url = medicine.get("bugiardino_url", "")
            if not source_url:
                raise ValueError("Per questo farmaco non è disponibile un bugiardino.")
            try:
                path = self.leaflet_store.save(
                    medicine["id"], fetch_aifa_pdf(source_url)
                )
            except ValueError as error:
                raise ValueError(f"Download del bugiardino non riuscito: {error}") from error

        from pypdf import PdfReader

        pages = PdfReader(str(path)).pages
        document_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        collection = self._vector_collection()
        where = {
            "$and": [
                {"medicine_id": medicine["id"]},
                {"document_hash": document_hash},
            ]
        }
        indexed = collection.get(where=where, limit=1)
        if not indexed["ids"]:
            chunks = []
            for page_number, page in enumerate(pages, start=1):
                text = " ".join((page.extract_text() or "").split())
                chunks.extend(
                    {"page": page_number, "text": part}
                    for part in self._split_text(text)
                )
            if not chunks:
                raise ValueError("Il PDF non contiene testo leggibile per la ricerca.")
            for offset in range(0, len(chunks), 64):
                batch = chunks[offset : offset + 64]
                embeddings = client.embeddings.create(
                    model=EMBEDDING_MODEL, input=[item["text"] for item in batch]
                )
                collection.upsert(
                    ids=[
                        f"{medicine['id']}-{document_hash}-{offset + index}"
                        for index in range(len(batch))
                    ],
                    documents=[item["text"] for item in batch],
                    embeddings=[item.embedding for item in embeddings.data],
                    metadatas=[
                        {
                            "medicine_id": medicine["id"],
                            "document_hash": document_hash,
                            "page": item["page"],
                        }
                        for item in batch
                    ],
                )

        query_embedding = client.embeddings.create(
            model=EMBEDDING_MODEL, input=[query]
        ).data[0].embedding
        result = collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where
        )
        return [
            {
                "id": medicine["nome"],
                "medicine_id": medicine["id"],
                "text": result["documents"][0][index],
                "page": result["metadatas"][0][index]["page"],
                "score": 1 - result["distances"][0][index],
            }
            for index in range(len(result["ids"][0]))
        ]

    def _find_mentioned_medicines(self, message: str) -> list[dict[str, Any]]:
        normalized_message = self._normalize_medicine_name(message)
        padded_message = f" {normalized_message} "
        matches = []
        for medicine in MedicineRepository(self.database).list():
            normalized_name = self._normalize_medicine_name(medicine["nome"])
            if normalized_name and f" {normalized_name} " in padded_message:
                matches.append(medicine)
        return sorted(
            matches,
            key=lambda medicine: padded_message.find(
                f" {self._normalize_medicine_name(medicine['nome'])} "
            ),
        )

    @staticmethod
    def _normalize_medicine_name(value: str) -> str:
        folded = unicodedata.normalize("NFKD", value.casefold())
        ascii_text = "".join(
            character
            for character in folded
            if not unicodedata.combining(character)
        )
        return " ".join(re.findall(r"[a-z0-9]+", ascii_text))

    @staticmethod
    def _split_text(text: str, size: int = 1_000, overlap: int = 150) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + size // 2, end)
                if boundary > start:
                    end = boundary
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
        return chunks

    def _condense(
        self,
        client: Any,
        model: str,
        history: list[dict[str, str]],
        message: str,
        medicine: str,
    ) -> str:
        if not history:
            return message
        conversation = "\n".join(
            f"{item['role']}: {item['content']}" for item in history[-12:]
        )
        result = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": (
                        "Riscrivi la nuova domanda in modo autonomo usando il contesto, "
                        "sostituendo pronomi e riferimenti vaghi. Mantieni il significato; "
                        "restituisci solo la domanda riscritta. Farmaco selezionato: "
                        f"{medicine}.\n\nConversazione:\n{conversation}\n\nDomanda:\n{message}"
                    ),
                }
            ],
        )
        return result.output_text.strip() or message

    @staticmethod
    def _generate(
        client: Any,
        model: str,
        chunks: list[dict[str, Any]],
        question: str,
        history: list[dict[str, str]],
    ) -> str:
        documents = "\n\n".join(
            f"<documento farmaco=\"{chunk['id']}\" pagina=\"{chunk['page']}\">\n"
            f"{chunk['text']}\n</documento>"
            for chunk in chunks
        )
        prompt = f"<documenti>\n{documents}\n</documenti>\n\n<domanda>\n{question}\n</domanda>"
        result = client.responses.create(
            model=model,
            instructions=INSTRUCTIONS,
            input=[
                {"role": item["role"], "content": item["content"]}
                for item in history
            ]
            + [{"role": "user", "content": prompt}],
        )
        return result.output_text.strip()

    def _summarize_if_needed(
        self,
        client: Any,
        model: str,
        history: list[dict[str, str]],
        budget: int,
    ) -> None:
        if self._count_tokens(history) <= budget or len(history) <= 2:
            return
        old_messages, last_turn = history[:-2], history[-2:]
        result = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": (
                        "Riassumi in poche righe la conversazione, preservando nomi dei "
                        "farmaci, fatti concreti e citazioni di pagina. Non aggiungere "
                        "informazioni.\n\n"
                        + "\n".join(
                            f"{item['role']}: {item['content']}" for item in old_messages
                        )
                    ),
                }
            ],
        )
        summary = result.output_text.strip()
        history[:] = [
            {"role": "user", "content": f"Riassunto della conversazione: {summary}"},
            {"role": "assistant", "content": "Ho presente il contesto riassunto."},
            *last_turn,
        ]
        while self._count_tokens(history) > budget and len(history) > 4:
            del history[2:4]

    @staticmethod
    def _truncate(history: list[dict[str, str]], budget: int) -> None:
        while len(history) > 2 and ChatService._count_tokens(history) > budget:
            del history[:2]

    @staticmethod
    def _count_tokens(messages: list[dict[str, str]]) -> int:
        try:
            import tiktoken

            encoder = tiktoken.get_encoding("o200k_base")
            return sum(len(encoder.encode(item["content"])) for item in messages)
        except ImportError:
            return sum(max(1, len(item["content"]) // 4) for item in messages)

    def _session_path(self, session_id: Any) -> Path:
        if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError("ID sessione non valido.")
        return self.sessions_dir / f"{session_id}.json"

    def _save_session(
        self, path: Path, messages: list[dict[str, str]], medicine_ids: list[str]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "messages": messages,
                "medicine_ids": medicine_ids,
                "medicine_id": medicine_ids[0] if medicine_ids else "",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f"{path.stem}-", delete=False
        ) as file:
            file.write(payload)
            temporary_path = Path(file.name)
        temporary_path.replace(path)