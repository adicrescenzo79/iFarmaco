import hashlib
import json
import logging
import os
import re
import tempfile
import unicodedata
from datetime import date
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
EMBEDDING_MODELS = ("text-embedding-3-small", "text-embedding-3-large")
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODELS[0]
MODEL_PRICES_USD_PER_MILLION = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
}
EMBEDDING_PRICES_USD_PER_MILLION = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}
MAX_MESSAGE_LENGTH = 4_000
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
INSTRUCTIONS = """Sei l'assistente informativo di iFarmaco. Per le domande cliniche
usa solo il contenuto dei foglietti illustrativi forniti nel turno corrente e
cita la fonte nel formato [Nome farmaco, p. N]. Se i documenti non contengono la
risposta, dichiaralo chiaramente e non inventare informazioni. Se la domanda non
nomina un farmaco, usa i passaggi pertinenti dell'intero corpus indicizzato e
cita ogni farmaco e pagina. Non formulare
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
    r"\b(scorte?|giacenze?|disponibilit[aà]|scadut[oi]|scadenz[ae])\b",
    re.IGNORECASE,
)
EXPIRY_QUESTION = re.compile(
    r"\bscadut[oi]\b", re.IGNORECASE
)
INVENTORY_QUANTITY = re.compile(
    r"\b(quante?|quanti)\b.*\b(ho|restano|rimangono|disponibili|"
    r"a disposizione)\b",
    re.IGNORECASE,
)
DOSAGE_FREQUENCY = re.compile(r"\b(quante?|quanti)\s+volte\b", re.IGNORECASE)
SYMPTOM_SEARCH_TERMS = {
    "mal di testa": ("mal di testa", "cefalea"),
    "cefalea": ("cefalea", "mal di testa"),
    "emicrania": ("emicrania",),
    "headache": ("headache", "cefalea", "mal di testa"),
}


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
        self._vector_client: Any = None
        self._collections: dict[str, Any] = {}

    @staticmethod
    def configuration() -> dict[str, Any]:
        return {
            "configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
            "models": list(CHAT_MODELS),
            "default_model": CHAT_MODELS[0],
            "embedding_models": list(EMBEDDING_MODELS),
            "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
        }

    @staticmethod
    def _record_usage(
        usage_records: list[dict[str, Any]] | None,
        model: str,
        response: Any,
        *,
        embeddings: bool = False,
    ) -> None:
        if usage_records is None:
            return
        usage = getattr(response, "usage", None)
        if usage is None:
            usage_records.append({"model": model, "usage_unavailable": True})
            return
        if embeddings:
            input_tokens = getattr(
                usage, "prompt_tokens", getattr(usage, "total_tokens", None)
            )
            output_tokens = 0
        else:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
        if (
            type(input_tokens) is not int
            or type(output_tokens) is not int
            or input_tokens < 0
            or output_tokens < 0
        ):
            usage_records.append({"model": model, "usage_unavailable": True})
            return
        usage_records.append(
            {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "embeddings": embeddings,
            }
        )

    @staticmethod
    def _estimate_interaction_cost(
        usage_records: list[dict[str, Any]],
    ) -> float | None:
        if any(record.get("usage_unavailable") for record in usage_records):
            return None
        cost = 0.0
        for record in usage_records:
            if record["embeddings"]:
                price = EMBEDDING_PRICES_USD_PER_MILLION[record["model"]]
                cost += record["input_tokens"] * price / 1_000_000
            else:
                input_price, output_price = MODEL_PRICES_USD_PER_MILLION[
                    record["model"]
                ]
                cost += (
                    record["input_tokens"] * input_price
                    + record["output_tokens"] * output_price
                ) / 1_000_000
        return round(cost, 8)

    def _response_details(
        self,
        *,
        show_reasoning: bool,
        reasoning: str,
        usage_records: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        details: dict[str, Any] = {}
        if show_reasoning and reasoning:
            details["reasoning"] = reasoning
        if usage_records is not None:
            cost = self._estimate_interaction_cost(usage_records)
            if cost is not None:
                details["interaction_cost_usd"] = cost
        return details

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
        if not isinstance(model, str) or model not in CHAT_MODELS:
            raise ValueError("Modello non supportato.")
        embedding_model = payload.get(
            "embedding_model", configuration["default_embedding_model"]
        )
        if (
            not isinstance(embedding_model, str)
            or embedding_model not in EMBEDDING_MODELS
        ):
            raise ValueError("Modello di ricerca non supportato.")
        show_reasoning = payload.get("show_reasoning", False)
        show_interaction_cost = payload.get("show_interaction_cost", False)
        if type(show_reasoning) is not bool or type(show_interaction_cost) is not bool:
            raise ValueError("Le preferenze di visualizzazione devono essere booleane.")
        usage_records = [] if show_interaction_cost else None
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
        if EXPIRY_QUESTION.search(message):
            expiry_medicines = mentioned or MedicineRepository(self.database).list()
            expired = self._expired_stock(expiry_medicines)
            if expired:
                details = []
                for medicine in expired:
                    lots = ", ".join(
                        f"{lot['nr_pezzi']} pezzi (scadenza "
                        f"{date.fromisoformat(lot['data_scadenza']).strftime('%d/%m/%Y')})"
                        for lot in medicine["lotti"]
                    )
                    details.append(f"- {medicine['nome']}: {lots}")
                answer = "Risultano queste scorte scadute:\n" + "\n".join(details)
            else:
                answer = "Non risultano scorte scadute tra i farmaci registrati."
            medicine_ids = [medicine["medicine_id"] for medicine in expired]
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "sources": [],
            }
            assistant_message.update(
                self._response_details(
                    show_reasoning=show_reasoning,
                    reasoning=(
                        "Ho controllato le date di scadenza dei lotti registrati "
                        "nel database locale."
                    ),
                    usage_records=usage_records,
                )
            )
            history.extend(
                [{"role": "user", "content": message}, assistant_message]
            )
            self._truncate(history, budget)
            self._save_session(session_path, history, medicine_ids)
            return {
                "session_id": session_id,
                "medicine_id": medicine_ids[0] if medicine_ids else "",
                "medicine_ids": medicine_ids,
                "medicine_names": [
                    medicine["nome"] for medicine in expired
                ],
                "answer": answer,
                "searched_query": message,
                "sources": [],
                "messages": history,
            }
        normalized_message = self._normalize_medicine_name(message)
        padded_message = f" {normalized_message} "
        has_supported_symptom = any(
            f" {symptom} " in padded_message for symptom in SYMPTOM_SEARCH_TERMS
        )
        if not medicines and not inventory_question and has_supported_symptom:
            symptom_search = self._search_available_leaflets(message)
            if symptom_search is not None:
                matches = symptom_search["matches"]
                if matches:
                    entries = "\n".join(
                        f"- {match['name']}: {match['pezzi_non_scaduti']} pezzi non scaduti; "
                        f"riferimento nel bugiardino a pagina {match['page']}."
                        for match in matches
                    )
                    answer = (
                        f"Nei bugiardini consultati ho trovato un riferimento a "
                        f"{symptom_search['symptom']} nella sezione sulle indicazioni per questi "
                        f"farmaci con scorte non scadute:\n{entries}\n"
                        "Questo elenco non stabilisce quale farmaco sia adatto a te e non è "
                        "un consiglio di assunzione. Verifica il foglietto e chiedi al medico "
                        "o al farmacista, soprattutto se assumi altri farmaci o hai patologie."
                    )
                else:
                    answer = (
                        f"Nei bugiardini consultabili non ho trovato un'indicazione d'uso "
                        f"che menzioni {symptom_search['symptom']} per farmaci con scorte "
                        "non scadute."
                    )
                if symptom_search["unavailable"]:
                    answer += (
                        f" Non ho potuto controllare {symptom_search['unavailable']} "
                        "bugiardini di farmaci disponibili: "
                        f"{symptom_search['missing_pdfs']} PDF mancanti e "
                        f"{symptom_search['unrecognized_sections']} sezioni non riconosciute."
                    )
                sources = [
                    {
                        "id": match["name"],
                        "medicine_id": match["medicine_id"],
                        "page": match["page"],
                        "score": 1.0,
                        "url": (
                            f"/api/medicines/{match['medicine_id']}/leaflet"
                            f"#page={match['page']}"
                        ),
                    }
                    for match in matches
                ]
                medicine_ids = [match["medicine_id"] for match in matches]
                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
                assistant_message.update(
                    self._response_details(
                        show_reasoning=show_reasoning,
                        reasoning=(
                            "Ho cercato riferimenti nelle sezioni delle indicazioni "
                            "dei bugiardini disponibili. Le pagine collegate mostrano "
                            "i passaggi usati; il confronto non indica quale farmaco "
                            "sia adatto alla persona."
                        ),
                        usage_records=usage_records,
                    )
                )
                history.extend(
                    [
                        {"role": "user", "content": message},
                        assistant_message,
                    ]
                )
                self._truncate(history, budget)
                self._save_session(session_path, history, medicine_ids)
                return {
                    "session_id": session_id,
                    "medicine_id": medicine_ids[0] if medicine_ids else "",
                    "medicine_ids": medicine_ids,
                    "medicine_names": [match["name"] for match in matches],
                    "answer": answer,
                    "searched_query": message,
                    "sources": sources,
                    "messages": history,
                    "needs_clarification": False,
                }
            answer = (
                "A quale farmaco ti riferisci? Puoi scriverne il nome nella domanda; "
                "per confrontarne più di uno, includi tutti i nomi."
            )
            history.extend(
                [
                    {"role": "user", "content": message},
                    {
                        "role": "assistant",
                        "content": answer,
                        **self._response_details(
                            show_reasoning=show_reasoning,
                            reasoning=(
                                "Non ho identificato con certezza il farmaco nella "
                                "domanda o nella conversazione. Chiedo un chiarimento "
                                "prima di consultare un bugiardino."
                            ),
                            usage_records=usage_records,
                        ),
                        "sources": [],
                    },
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
                usage_records=usage_records,
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
            reasoning = (
                "Ho verificato le giacenze nel database locale"
                + (
                    f" per {', '.join(medicine_names)}."
                    if medicine_names
                    else " per i farmaci individuati nella domanda."
                )
            )
        else:
            medicine_names = [medicine["nome"] for medicine in medicines]
            search_scope = ", ".join(medicine_names) or "tutti i bugiardini indicizzati"
            condense_arguments = (
                {"usage_records": usage_records}
                if usage_records is not None
                else {}
            )
            searched_query = self._condense(
                client,
                model,
                history,
                message,
                search_scope,
                **condense_arguments,
            )
            chunks = []
            if medicines:
                for medicine in medicines:
                    chunks.extend(
                        self._retrieve(
                            client,
                            medicine,
                            searched_query,
                            top_k,
                            embedding_model,
                            usage_records=usage_records,
                        )
                    )
            else:
                chunks = self._retrieve_all(
                    client,
                    searched_query,
                    top_k,
                    embedding_model,
                    usage_records=usage_records,
                )
            chunks.sort(key=lambda chunk: chunk["score"], reverse=True)
            chunks = chunks[:top_k]
            answer = (
                self._generate(
                    client,
                    model,
                    chunks,
                    searched_query,
                    history,
                    usage_records=usage_records,
                )
                if chunks
                else "Non trovo informazioni pertinenti nei bugiardini indicizzati."
            )
            if medicines:
                medicine_ids = [medicine["id"] for medicine in medicines]
                reasoning = (
                    f"Ho cercato “{searched_query}” nel bugiardino di "
                    f"{', '.join(medicine_names)} e ho selezionato {len(chunks)} "
                    "passaggi per pertinenza. Puoi verificare le pagine nelle fonti."
                )
            else:
                medicine_ids = list(
                    dict.fromkeys(chunk["medicine_id"] for chunk in chunks)
                )
                medicine_names = list(dict.fromkeys(chunk["id"] for chunk in chunks))
                reasoning = (
                    f"Ho cercato “{searched_query}” in tutti i bugiardini indicizzati "
                    f"e selezionato {len(chunks)} passaggi per pertinenza."
                )
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
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
        assistant_message.update(
            self._response_details(
                show_reasoning=show_reasoning,
                reasoning=reasoning,
                usage_records=None,
            )
        )
        history.extend(
            [{"role": "user", "content": message}, assistant_message]
        )
        if memory_mode == "summary":
            self._summarize_if_needed(
                client, model, history, budget, usage_records=usage_records
            )
        else:
            self._truncate(history, budget)
        if usage_records is not None:
            cost = self._estimate_interaction_cost(usage_records)
            if cost is not None:
                assistant_message["interaction_cost_usd"] = cost
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
        usage_records: list[dict[str, Any]] | None = None,
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
            self._record_usage(usage_records, model, response)
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
        self._record_usage(usage_records, model, response)
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

    def _expired_stock(
        self, medicines: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        stock = StockRepository(self.database)
        expired_medicines = []
        for medicine in medicines:
            lots = [
                lot
                for lot in stock.for_medicine(medicine["id"])
                if not lot["non_scaduto"]
            ]
            if lots:
                expired_medicines.append(
                    {
                        "medicine_id": medicine["id"],
                        "nome": medicine["nome"],
                        "lotti": lots,
                    }
                )
        return expired_medicines

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

    def _search_available_leaflets(self, message: str) -> dict[str, Any] | None:
        normalized_message = self._normalize_medicine_name(message)
        padded_message = f" {normalized_message} "
        symptom = next(
            (
                label
                for label in SYMPTOM_SEARCH_TERMS
                if f" {label} " in padded_message
            ),
            None,
        )
        if symptom is None:
            return None

        from pypdf import PdfReader

        search_terms = SYMPTOM_SEARCH_TERMS[symptom]
        stock = StockRepository(self.database)
        matches = []
        missing_pdfs = 0
        unrecognized_sections = 0
        for medicine in MedicineRepository(self.database).list():
            lots = stock.for_medicine(medicine["id"])
            available_pieces = sum(
                lot["nr_pezzi"] for lot in lots if lot["non_scaduto"]
            )
            if available_pieces <= 0:
                continue

            path = self.leaflet_store.find(medicine["id"])
            if path is None:
                missing_pdfs += 1
                continue
            page_texts = [
                self._normalize_medicine_name(page.extract_text() or "")
                for page in PdfReader(str(path)).pages
            ]
            section = self._indication_section(page_texts)
            if section is None:
                unrecognized_sections += 1
                continue
            section_text, section_start = section
            term_matches = [
                (match.start(), term)
                for term in search_terms
                if (match := re.search(rf"\b{re.escape(term)}\b", section_text))
            ]
            if not term_matches:
                continue

            term_offset, _ = min(term_matches)
            absolute_offset = section_start + term_offset
            page_number = 1
            page_offset = 0
            for index, page_text in enumerate(page_texts, start=1):
                if absolute_offset <= page_offset + len(page_text):
                    page_number = index
                    break
                page_offset += len(page_text) + 1
            matches.append(
                {
                    "medicine_id": medicine["id"],
                    "name": medicine["nome"],
                    "pezzi_non_scaduti": available_pieces,
                    "page": page_number,
                }
            )

        return {
            "symptom": symptom,
            "matches": matches,
            "missing_pdfs": missing_pdfs,
            "unrecognized_sections": unrecognized_sections,
            "unavailable": missing_pdfs + unrecognized_sections,
        }

    @staticmethod
    def _indication_section(page_texts: list[str]) -> tuple[str, int] | None:
        full_text = "\n".join(page_texts)
        headings = [
            match.end()
            for match in re.finditer(
                r"\b(?:a (?:che )?cosa|e cosa) serve\b", full_text
            )
        ]
        sections = []
        for start in headings:
            end = re.search(
                r"\b2\s+(?:cosa deve sapere|cosa deve fare|prima di prendere)\b",
                full_text[start:],
            )
            if end:
                section_end = start + end.start()
                if section_end > start:
                    sections.append((full_text[start:section_end], start))
        if not sections:
            starts = [
                match.end()
                for match in re.finditer(r"\bindicazioni terapeutiche\b", full_text)
            ]
            for start in starts:
                end = re.search(
                    r"\b(?:controindicazioni|posologia|modalita di somministrazione|"
                    r"avvertenze|effetti indesiderati|interazioni)\b",
                    full_text[start:],
                )
                if end:
                    section_end = start + end.start()
                    if section_end > start:
                        sections.append((full_text[start:section_end], start))
        return max(sections, key=lambda section: len(section[0])) if sections else None

    def _openai_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def _vector_collection(
        self, embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ) -> Any:
        if embedding_model not in self._collections:
            import chromadb

            if self._vector_client is None:
                self.chroma_dir.mkdir(parents=True, exist_ok=True)
                self._vector_client = chromadb.PersistentClient(
                    path=str(self.chroma_dir)
                )
            collection_name = (
                "ifarmaco_leaflets"
                if embedding_model == DEFAULT_EMBEDDING_MODEL
                else f"ifarmaco_leaflets_{embedding_model.replace('-', '_')}"
            )
            collection = self._vector_client.get_or_create_collection(
                collection_name, metadata={"hnsw:space": "cosine"}
            )
            self._collections[embedding_model] = collection
        return self._collections[embedding_model]

    def index_all_medicines(
        self, embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ) -> tuple[int, int, list[str]]:
        if embedding_model not in EMBEDDING_MODELS:
            raise ValueError("Modello di ricerca non supportato.")
        if not self.configuration()["configured"]:
            raise ChatUnavailableError("Configura OPENAI_API_KEY nel file .env.")

        medicines = MedicineRepository(self.database).list()
        client = self._openai_client()
        newly_indexed = 0
        failures = []
        for position, medicine in enumerate(medicines, start=1):
            logging.info(
                "Indicizzazione bugiardino %d/%d: %s",
                position,
                len(medicines),
                medicine["id"],
            )
            try:
                _, created = self._ensure_medicine_indexed(
                    client, medicine, embedding_model
                )
                newly_indexed += created
            except Exception as error:
                failures.append(f"{medicine['id']}: {error}")
        return len(medicines), newly_indexed, failures

    def index_medicine(self, medicine_id: str, embedding_model: str) -> None:
        if embedding_model not in EMBEDDING_MODELS:
            raise ValueError("Modello di ricerca non supportato.")
        if not self.configuration()["configured"]:
            raise ChatUnavailableError("Configura OPENAI_API_KEY nel file .env.")
        medicine = MedicineRepository(self.database).get(medicine_id)
        self._ensure_medicine_indexed(
            self._openai_client(), medicine, embedding_model
        )

    def _ensure_medicine_indexed(
        self,
        client: Any,
        medicine: dict[str, Any],
        embedding_model: str,
        usage_records: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], bool]:
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
        where = {
            "$and": [
                {"medicine_id": medicine["id"]},
                {"document_hash": document_hash},
            ]
        }
        collection = self._vector_collection(embedding_model)
        chunks = []
        for page_number, page in enumerate(pages, start=1):
            text = " ".join((page.extract_text() or "").split())
            chunks.extend(
                {"page": page_number, "text": part}
                for part in self._split_text(text)
            )
        if not chunks:
            raise ValueError("Il PDF non contiene testo leggibile per la ricerca.")
        expected_ids = {
            f"{medicine['id']}-{document_hash}-{index}"
            for index in range(len(chunks))
        }
        indexed_ids = set(collection.get(where=where)["ids"])
        if expected_ids.issubset(indexed_ids):
            return where, False
        for offset in range(0, len(chunks), 64):
            batch = chunks[offset : offset + 64]
            embeddings = client.embeddings.create(
                model=embedding_model, input=[item["text"] for item in batch]
            )
            self._record_usage(
                usage_records, embedding_model, embeddings, embeddings=True
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
        return where, True

    def _retrieve(
        self,
        client: Any,
        medicine: dict[str, Any],
        query: str,
        top_k: int,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        usage_records: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        where, _ = self._ensure_medicine_indexed(
            client, medicine, embedding_model, usage_records
        )
        collection = self._vector_collection(embedding_model)
        query_embedding = client.embeddings.create(
            model=embedding_model, input=[query]
        )
        self._record_usage(
            usage_records, embedding_model, query_embedding, embeddings=True
        )
        result = collection.query(
            query_embeddings=[query_embedding.data[0].embedding],
            n_results=top_k,
            where=where,
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

    def _retrieve_all(
        self,
        client: Any,
        query: str,
        top_k: int,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        usage_records: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        collection = self._vector_collection(embedding_model)
        count = collection.count()
        if count == 0:
            return []

        query_embedding = client.embeddings.create(
            model=embedding_model, input=[query]
        )
        self._record_usage(
            usage_records, embedding_model, query_embedding, embeddings=True
        )
        result = collection.query(
            query_embeddings=[query_embedding.data[0].embedding],
            n_results=min(top_k, count),
        )
        medicine_names = {
            medicine["id"]: medicine["nome"]
            for medicine in MedicineRepository(self.database).list()
        }
        return [
            {
                "id": medicine_names.get(
                    result["metadatas"][0][index]["medicine_id"],
                    result["metadatas"][0][index]["medicine_id"],
                ),
                "medicine_id": result["metadatas"][0][index]["medicine_id"],
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
        usage_records: list[dict[str, Any]] | None = None,
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
        self._record_usage(usage_records, model, result)
        return result.output_text.strip() or message

    @staticmethod
    def _generate(
        client: Any,
        model: str,
        chunks: list[dict[str, Any]],
        question: str,
        history: list[dict[str, str]],
        usage_records: list[dict[str, Any]] | None = None,
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
        ChatService._record_usage(usage_records, model, result)
        return result.output_text.strip()

    def _summarize_if_needed(
        self,
        client: Any,
        model: str,
        history: list[dict[str, str]],
        budget: int,
        usage_records: list[dict[str, Any]] | None = None,
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
        self._record_usage(usage_records, model, result)
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