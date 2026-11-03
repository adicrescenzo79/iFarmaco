import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.chat_service import ChatService
from backend.database import Database
from backend.leaflets import LeafletStore
from backend.repositories import MedicineRepository, StockRepository


class ChatServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database = Database(root / "chat.sqlite3")
        self.database.initialize()
        MedicineRepository(self.database).create(
            {
                "id": "00011",
                "nome": "Farmaco di prova",
                "bugiardino_url": "",
            }
        )
        MedicineRepository(self.database).create(
            {
                "id": "00013",
                "nome": "AMOXICILLINA E ACIDO CLAVULANICO ABC",
                "bugiardino_url": "",
            }
        )
        stock = StockRepository(self.database)
        stock.create(
            {
                "farmaco_id": "00013",
                "data_scadenza": "2027-01-31",
                "nr_pezzi": 12,
            }
        )
        stock.create(
            {
                "farmaco_id": "00013",
                "data_scadenza": "2025-01-31",
                "nr_pezzi": 2,
            }
        )
        MedicineRepository(self.database).create(
            {
                "id": "00012",
                "nome": "Altro farmaco",
                "bugiardino_url": "",
            }
        )
        self.service = ChatService(
            self.database,
            leaflet_store=LeafletStore(root / "bugiardini"),
            sessions_dir=root / "sessions",
            chroma_dir=root / "chroma",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_chat_turn_and_follow_up_are_saved_with_sources(self) -> None:
        client = Mock()
        chunk = {
            "id": "Farmaco di prova",
            "medicine_id": "00011",
            "page": 2,
            "score": 0.91,
            "text": "Testo",
        }
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(self.service, "_openai_client", return_value=client),
            patch.object(self.service, "_retrieve", return_value=[chunk]),
            patch.object(self.service, "_condense", side_effect=lambda _, __, ___, query, ____: query),
            patch.object(self.service, "_generate", return_value="Risposta documentata.") as generate,
            patch.object(ChatService, "_count_tokens", return_value=10),
        ):
            first = self.service.respond(
                {
                    "session_id": "test-session",
                    "medicine_id": "00011",
                    "message": "A cosa serve?",
                }
            )
            second = self.service.respond(
                {
                    "session_id": "test-session",
                    "medicine_id": "00011",
                    "message": "E gli effetti indesiderati?",
                }
            )

        self.assertEqual(first["answer"], "Risposta documentata.")
        self.assertEqual(first["sources"][0]["page"], 2)
        self.assertEqual(len(second["messages"]), 4)
        self.assertEqual(generate.call_count, 2)
        saved = self.service.session("test-session")
        self.assertEqual(saved["medicine_id"], "00011")
        self.assertEqual(saved["messages"][-1]["sources"][0]["page"], 2)

    def test_summary_memory_replaces_older_turns(self) -> None:
        history = [
            {"role": "user", "content": "Domanda vecchia"},
            {"role": "assistant", "content": "Risposta vecchia"},
            {"role": "user", "content": "Domanda recente"},
            {"role": "assistant", "content": "Risposta recente"},
        ]
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(output_text="Farmaco e contesto")
        with patch.object(ChatService, "_count_tokens", side_effect=[200, 80]):
            self.service._summarize_if_needed(client, "gpt-4o-mini", history, 150)

        self.assertEqual(len(history), 4)
        self.assertIn("Farmaco e contesto", history[0]["content"])
        self.assertEqual(history[-2]["content"], "Domanda recente")

    def test_multiple_medicine_names_are_retrieved_without_selection(self) -> None:
        client = Mock()
        chunks = [
            {
                "id": name,
                "medicine_id": medicine_id,
                "page": 1,
                "score": score,
                "text": f"Testo {name}",
            }
            for name, medicine_id, score in (
                ("Farmaco di prova", "00011", 0.9),
                ("Altro farmaco", "00012", 0.8),
            )
        ]
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(self.service, "_openai_client", return_value=client),
            patch.object(self.service, "_retrieve", side_effect=[[chunks[0]], [chunks[1]]]) as retrieve,
            patch.object(self.service, "_condense", side_effect=lambda _, __, ___, query, ____: query),
            patch.object(self.service, "_generate", return_value="Confronto documentato."),
            patch.object(ChatService, "_count_tokens", return_value=10),
        ):
            result = self.service.respond(
                {
                    "session_id": "multi-session",
                    "medicine_id": None,
                    "message": "Confronta Farmaco di prova e Altro farmaco",
                }
            )

        self.assertEqual(result["medicine_ids"], ["00011", "00012"])
        self.assertEqual({source["medicine_id"] for source in result["sources"]}, {"00011", "00012"})
        self.assertEqual(retrieve.call_count, 2)

    def test_asks_for_medicine_when_none_is_named_or_in_session(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(self.service, "_openai_client") as openai_client,
            patch.object(ChatService, "_count_tokens", return_value=10),
        ):
            result = self.service.respond(
                {
                    "session_id": "clarify-session",
                    "message": "A cosa serve?",
                }
            )

        self.assertTrue(result["needs_clarification"])
        self.assertIn("A quale farmaco", result["answer"])
        openai_client.assert_not_called()

    def test_inventory_tool_returns_expired_and_valid_lots_separately(self) -> None:
        result = self.service._execute_inventory_tool(
            {"medicine_name": "Ximocillina"}
        )

        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"][0]["medicine_id"], "00013")
        self.assertEqual(result["matches"][0]["pezzi_non_scaduti"], 12)
        self.assertEqual(result["matches"][0]["pezzi_scaduti"], 2)

    def test_inventory_intent_does_not_capture_dosage_frequency(self) -> None:
        self.assertTrue(ChatService._is_inventory_question("Quante Ximocillina ho?"))
        self.assertFalse(
            ChatService._is_inventory_question(
                "Quante volte al giorno devo assumere Ximocillina?"
            )
        )

    def test_inventory_request_runs_function_call_and_returns_model_answer(self) -> None:
        call = SimpleNamespace(
            type="function_call",
            name="get_medicine_stock",
            arguments=json.dumps({"medicine_name": "Ximocillina"}),
            call_id="call-stock-1",
        )
        client = Mock()
        client.responses.create.side_effect = [
            SimpleNamespace(output=[call], output_text=""),
            SimpleNamespace(output=[], output_text="Hai 12 pezzi non scaduti."),
        ]

        answer, tool_results = self.service._answer_inventory(
            client,
            "gpt-4o-mini",
            [],
            "Quante Ximocillina ho?",
            [],
        )

        self.assertEqual(answer, "Hai 12 pezzi non scaduti.")
        self.assertEqual(tool_results[0]["matches"][0]["pezzi_non_scaduti"], 12)
        self.assertEqual(client.responses.create.call_count, 2)
        second_input = client.responses.create.call_args_list[1].kwargs["input"]
        self.assertEqual(second_input[-1]["type"], "function_call_output")

    def test_inventory_question_routes_to_tool_without_leaflet_retrieval(self) -> None:
        tool_results = [
            {
                "matches": [
                    {
                        "medicine_id": "00013",
                        "name": "AMOXICILLINA E ACIDO CLAVULANICO ABC",
                    }
                ]
            }
        ]
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(self.service, "_openai_client", return_value=Mock()),
            patch.object(
                self.service,
                "_answer_inventory",
                return_value=("Hai 12 pezzi non scaduti.", tool_results),
            ),
            patch.object(self.service, "_retrieve") as retrieve,
            patch.object(ChatService, "_count_tokens", return_value=10),
        ):
            result = self.service.respond(
                {
                    "session_id": "inventory-session",
                    "message": "Quante Ximocillina ho?",
                }
            )

        self.assertEqual(result["medicine_ids"], ["00013"])
        self.assertEqual(result["answer"], "Hai 12 pezzi non scaduti.")
        retrieve.assert_not_called()

    def test_session_id_cannot_escape_storage_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "ID sessione"):
            self.service.session("../outside")


if __name__ == "__main__":
    unittest.main()