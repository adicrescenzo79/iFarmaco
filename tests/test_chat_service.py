import os
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.chat_service import INSTRUCTIONS, ChatService
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

    def test_instructions_exclude_unsolicited_personal_habit_advice(self) -> None:
        self.assertIn("non aggiungere consigli generici o non", INSTRUCTIONS)
        self.assertIn("non presumere né", INSTRUCTIONS)
        self.assertIn("direttamente pertinenti alla domanda sul farmaco", INSTRUCTIONS)

    def test_embedding_model_uses_its_own_persistent_index(self) -> None:
        vector_client = Mock()
        vector_client.get_or_create_collection.side_effect = ["small", "large"]
        chromadb = SimpleNamespace(
            PersistentClient=Mock(return_value=vector_client)
        )
        with patch.dict("sys.modules", {"chromadb": chromadb}):
            small = self.service._vector_collection("text-embedding-3-small")
            large = self.service._vector_collection("text-embedding-3-large")
            small_again = self.service._vector_collection("text-embedding-3-small")

        self.assertEqual((small, large, small_again), ("small", "large", "small"))
        self.assertEqual(
            [
                call.args[0]
                for call in vector_client.get_or_create_collection.call_args_list
            ],
            [
                "ifarmaco_leaflets_text_embedding_3_small",
                "ifarmaco_leaflets_text_embedding_3_large",
            ],
        )

    def test_global_retrieval_queries_without_medicine_filter(self) -> None:
        collection = Mock()
        collection.count.return_value = 3
        collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Testo del bugiardino."]],
            "metadatas": [[{"medicine_id": "00011", "page": 2}]],
            "distances": [[0.1]],
        }
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])]
        )
        with patch.object(self.service, "_vector_collection", return_value=collection):
            chunks = self.service._retrieve_all(
                client, "paracetamolo", 4, "text-embedding-3-small"
            )

        self.assertEqual(chunks[0]["id"], "Farmaco di prova")
        self.assertEqual(chunks[0]["page"], 2)
        self.assertEqual(collection.query.call_args.kwargs["n_results"], 3)
        self.assertNotIn("where", collection.query.call_args.kwargs)

    def test_hybrid_retrieval_promotes_exact_keyword_match(self) -> None:
        collection = Mock()
        collection.count.return_value = 3
        collection.get.return_value = {
            "ids": ["semantic", "exact", "other"],
            "documents": [
                "La terapia tratta infezioni batteriche.",
                "Il codice AIC 12345 è riportato nel foglietto.",
                "Informazioni generali del medicinale.",
            ],
            "metadatas": [
                {"medicine_id": "00011", "page": 1},
                {"medicine_id": "00012", "page": 2},
                {"medicine_id": "00013", "page": 3},
            ],
        }
        collection.query.return_value = {
            "ids": [["semantic", "exact", "other"]],
            "documents": [[
                "La terapia tratta infezioni batteriche.",
                "Il codice AIC 12345 è riportato nel foglietto.",
                "Informazioni generali del medicinale.",
            ]],
            "metadatas": [[
                {"medicine_id": "00011", "page": 1},
                {"medicine_id": "00012", "page": 2},
                {"medicine_id": "00013", "page": 3},
            ]],
            "distances": [[0.1, 0.2, 0.3]],
        }
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])]
        )

        with patch.object(self.service, "_vector_collection", return_value=collection):
            chunks = self.service._retrieve_all(
                client,
                "Qual è il codice AIC 12345?",
                1,
                search_mode="hybrid",
            )

        self.assertEqual(chunks[0]["chunk_id"], "exact")
        self.assertEqual(chunks[0]["medicine_id"], "00012")
        self.assertLess(chunks[0]["score"], 1)

    def test_chat_uses_selected_hybrid_retrieval_mode(self) -> None:
        chunk = {
            "id": "Farmaco di prova",
            "medicine_id": "00011",
            "page": 1,
            "score": 0.02,
            "text": "Testo del bugiardino.",
        }
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(self.service, "_openai_client", return_value=Mock()),
            patch.object(
                self.service, "_retrieve_all", return_value=[chunk]
            ) as retrieve,
            patch.object(
                self.service, "_condense", side_effect=lambda _, __, ___, query, ____: query
            ),
            patch.object(
                self.service, "_generate", return_value="Risposta documentata."
            ),
            patch.object(ChatService, "_count_tokens", return_value=10),
        ):
            result = self.service.respond(
                {
                    "session_id": "hybrid-session",
                    "message": "Informazioni sul farmaco",
                    "retrieval_mode": "hybrid",
                }
            )

        self.assertEqual(result["answer"], "Risposta documentata.")
        self.assertEqual(retrieve.call_args.kwargs["search_mode"], "hybrid")

    def test_diversified_retrieval_scans_all_chunks_and_excludes_prior_medicines(
        self,
    ) -> None:
        collection = Mock()
        collection.count.return_value = 4
        collection.query.return_value = {
            "ids": [["chunk-1", "chunk-2", "chunk-3", "chunk-4"]],
            "documents": [["A", "B", "C", "D"]],
            "metadatas": [[
                {"medicine_id": "00013", "page": 1},
                {"medicine_id": "00013", "page": 2},
                {"medicine_id": "00014", "page": 3},
                {"medicine_id": "00011", "page": 4},
            ]],
            "distances": [[0.1, 0.2, 0.3, 0.4]],
        }
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])]
        )

        with patch.object(self.service, "_vector_collection", return_value=collection):
            chunks = self.service._retrieve_all(
                client,
                "dolore addominale",
                4,
                diversify_by_medicine=True,
                eligible_medicine_ids={"00013", "00014"},
                excluded_medicine_ids={"00013"},
            )

        self.assertEqual([chunk["medicine_id"] for chunk in chunks], ["00014"])
        self.assertEqual(collection.query.call_args.kwargs["n_results"], 4)

    def test_document_embeddings_are_created_in_batches_of_64(self) -> None:
        pdf_path = Path(self.temp_dir.name) / "fixture.pdf"
        pdf_path.write_bytes(b"fixture")
        page = SimpleNamespace(extract_text=lambda: "Testo del bugiardino.")
        collection = Mock()
        document_hash = hashlib.sha256(b"fixture").hexdigest()[:16]
        collection.get.return_value = {
            "ids": [f"00011-{document_hash}-0"]
        }
        client = Mock()
        client.embeddings.create.side_effect = lambda *, model, input: SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0]) for _ in input]
        )

        with (
            patch.object(self.service.leaflet_store, "find", return_value=pdf_path),
            patch("pypdf.PdfReader", return_value=SimpleNamespace(pages=[page] * 65)),
            patch.object(self.service, "_vector_collection", return_value=collection),
        ):
            _, created = self.service._ensure_medicine_indexed(
                client,
                {"id": "00011", "bugiardino_url": ""},
                "text-embedding-3-small",
            )

        self.assertTrue(created)
        self.assertEqual(
            [len(call.kwargs["input"]) for call in client.embeddings.create.call_args_list],
            [64, 1],
        )
        self.assertEqual(collection.upsert.call_count, 2)

    def test_usage_estimate_includes_response_and_embedding_tokens(self) -> None:
        usage_records = []
        ChatService._record_usage(
            usage_records,
            "gpt-4o-mini",
            SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1_000, output_tokens=500)
            ),
        )
        ChatService._record_usage(
            usage_records,
            "text-embedding-3-small",
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1_000)),
            embeddings=True,
        )

        self.assertEqual(
            ChatService._estimate_interaction_cost(usage_records), 0.00047
        )

    def test_streamed_generation_delivers_text_deltas(self) -> None:
        completed_response = SimpleNamespace(
            output_text="Risposta documentata.", usage=None
        )
        client = Mock()
        client.responses.create.return_value = iter(
            [
                SimpleNamespace(
                    type="response.output_text.delta", delta="Risposta "
                ),
                SimpleNamespace(
                    type="response.output_text.delta", delta="documentata."
                ),
                SimpleNamespace(
                    type="response.completed", response=completed_response
                ),
            ]
        )
        deltas = []

        answer = ChatService._generate(
            client,
            "gpt-4o-mini",
            [{"id": "Farmaco di prova", "page": 2, "text": "Testo"}],
            "A cosa serve?",
            [],
            on_delta=deltas.append,
        )

        self.assertEqual(answer, "Risposta documentata.")
        self.assertEqual(deltas, ["Risposta ", "documentata."])
        self.assertTrue(client.responses.create.call_args.kwargs["stream"])

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
        self.assertEqual(first["sources"][0]["excerpt"], "Testo")
        self.assertNotIn("reasoning", first["messages"][-1])
        self.assertEqual(len(second["messages"]), 4)
        self.assertEqual(generate.call_count, 2)
        saved = self.service.session("test-session")
        self.assertEqual(saved["medicine_id"], "00011")
        self.assertEqual(saved["messages"][-1]["sources"][0]["page"], 2)
        self.assertEqual(saved["messages"][-1]["sources"][0]["excerpt"], "Testo")

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

    def test_question_without_medicine_searches_global_index(self) -> None:
        chunk = {
            "id": "PARACETAMOLO E CODEINA EG",
            "medicine_id": "00011",
            "page": 2,
            "score": 0.91,
            "text": "Contenuto del bugiardino.",
        }
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(self.service, "_openai_client", return_value=Mock()),
            patch.object(self.service, "_retrieve_all", return_value=[chunk]) as retrieve,
            patch.object(
                self.service, "_generate", return_value="Risposta dai bugiardini."
            ),
            patch.object(ChatService, "_count_tokens", return_value=10),
        ):
            result = self.service.respond(
                {
                    "session_id": "clarify-session",
                    "message": "Ho un farmaco a base di paracetamolo?",
                }
            )

        self.assertNotIn("needs_clarification", result)
        self.assertEqual(result["answer"], "Risposta dai bugiardini.")
        self.assertEqual(result["medicine_ids"], ["00011"])
        self.assertEqual(result["sources"][0]["page"], 2)
        retrieve.assert_called_once()

    def test_symptom_question_searches_stocked_leaflet_indications(self) -> None:
        page = SimpleNamespace(
            extract_text=lambda: (
                "1. Cos'e Farmaco di prova e cosa serve. "
                "Indicato per il trattamento del mal di testa. "
                "2. Cosa deve fare prima di prendere questo medicinale."
            )
        )
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(self.service, "_openai_client") as openai_client,
            patch.object(
                self.service.leaflet_store,
                "find",
                return_value=Path(self.temp_dir.name) / "00013.pdf",
            ),
            patch("pypdf.PdfReader", return_value=SimpleNamespace(pages=[page])),
        ):
            result = self.service.respond(
                {
                    "session_id": "headache-session",
                    "message": "Cosa posso prendere per il mal di testa?",
                }
            )

        self.assertFalse(result["needs_clarification"])
        self.assertEqual(result["medicine_ids"], ["00013"])
        self.assertEqual(result["sources"][0]["page"], 1)
        self.assertIn("mal di testa", result["sources"][0]["excerpt"])
        self.assertIn("12 pezzi non scaduti", result["answer"])
        self.assertIn("non è un consiglio di assunzione", result["answer"])
        openai_client.assert_not_called()

    def test_source_excerpt_is_limited_and_keeps_relevant_context(self) -> None:
        text = f"Informazioni iniziali. {'contenuto ' * 60} principio attivo "
        excerpt = self.service._source_excerpt(
            text, max_length=80, matched_term="principio attivo"
        )

        self.assertLessEqual(len(excerpt), 80)
        self.assertIn("principio attivo", excerpt)
        self.assertTrue(excerpt.startswith("…"))

    def test_alternative_follow_up_searches_other_medicines_globally(self) -> None:
        repository = MedicineRepository(self.database)
        repository.create({"id": "00014", "nome": "Altro farmaco pertinente"})
        StockRepository(self.database).create(
            {
                "farmaco_id": "00014",
                "data_scadenza": "2099-12-31",
                "nr_pezzi": 3,
            }
        )
        self.service._save_session(
            self.service._session_path("general-alternatives"),
            [
                {
                    "role": "user",
                    "content": "Cosa posso prendere per il mal di pancia?",
                },
                {
                    "role": "assistant",
                    "content": "Il foglietto del Farmaco di prova riporta indicazioni.",
                    "sources": [
                        {
                            "medicine_id": "00011",
                            "id": "Farmaco di prova",
                            "page": 2,
                        }
                    ],
                },
            ],
            ["00011"],
        )
        other_chunk = {
            "id": "Altro farmaco pertinente",
            "medicine_id": "00014",
            "page": 3,
            "score": 0.88,
            "text": "Testo pertinente del foglietto.",
        }

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(self.service, "_openai_client") as openai_client,
            patch.object(
                self.service,
                "_retrieve_all",
                return_value=[other_chunk],
            ) as retrieve_all,
            patch.object(self.service, "_retrieve") as retrieve_one,
            patch.object(self.service, "_condense", return_value="mal di pancia"),
            patch.object(
                self.service,
                "_generate",
                return_value="Ho trovato un altro farmaco pertinente.",
            ),
            patch.object(ChatService, "_count_tokens", return_value=10),
        ):
            result = self.service.respond(
                {
                    "session_id": "general-alternatives",
                    "message": "E non ce ne sono altri?",
                }
            )

        self.assertEqual(result["medicine_ids"], ["00014"])
        self.assertEqual(result["medicine_names"], ["Altro farmaco pertinente"])
        self.assertEqual(result["sources"][0]["medicine_id"], "00014")
        self.assertEqual(retrieve_all.call_args.kwargs["eligible_medicine_ids"], {"00013", "00014"})
        self.assertEqual(retrieve_all.call_args.kwargs["excluded_medicine_ids"], {"00011"})
        self.assertTrue(retrieve_all.call_args.kwargs["diversify_by_medicine"])
        retrieve_one.assert_not_called()
        openai_client.assert_called_once()

    def test_alternative_follow_up_detection_is_topic_agnostic(self) -> None:
        prior_hiv_answer = [
            {
                "role": "assistant",
                "content": "Ho trovato Truvada.",
                "sources": [{"medicine_id": "00014"}],
            }
        ]
        prior_symptom_answer = [
            {
                "role": "assistant",
                "content": "Il foglietto menziona Farmaco di prova.",
                "sources": [{"medicine_id": "00011"}],
            }
        ]

        self.assertTrue(
            self.service._is_broad_medicine_follow_up(
                "E non ce ne sono altri?", prior_hiv_answer
            )
        )
        self.assertTrue(
            self.service._is_broad_medicine_follow_up(
                "A parte il Farmaco di prova", prior_symptom_answer
            )
        )
        self.assertFalse(
            self.service._is_broad_medicine_follow_up(
                "Altra domanda", prior_hiv_answer
            )
        )

    def test_self_contained_inventory_question_resets_previous_medicine_context(
        self,
    ) -> None:
        self.service._save_session(
            self.service._session_path("topic-change-inventory"),
            [
                {"role": "user", "content": "Per l'HIV c'è qualcosa?"},
                {
                    "role": "assistant",
                    "content": "Ho trovato Truvada.",
                    "sources": [{"medicine_id": "00011", "id": "Truvada", "page": 2}],
                },
            ],
            ["00011"],
        )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(self.service, "_openai_client", return_value=Mock()),
            patch.object(
                self.service,
                "_answer_inventory",
                return_value=("Hai antibiotici disponibili.", []),
            ) as answer_inventory,
            patch.object(self.service, "_retrieve") as retrieve_one,
        ):
            result = self.service.respond(
                {
                    "session_id": "topic-change-inventory",
                    "message": "Ho degli antibiotici?",
                }
            )

        self.assertEqual(result["answer"], "Hai antibiotici disponibili.")
        self.assertEqual(answer_inventory.call_args.args[2], [])
        self.assertEqual(answer_inventory.call_args.args[4], [])
        retrieve_one.assert_not_called()
        self.assertTrue(
            ChatService._is_inventory_question(
                "Ho degli antibiotici?"
            )
        )

    def test_self_contained_new_topic_searches_globally_without_old_history(
        self,
    ) -> None:
        self.service._save_session(
            self.service._session_path("topic-change-symptom"),
            [
                {"role": "user", "content": "Per l'HIV c'è qualcosa?"},
                {
                    "role": "assistant",
                    "content": "Ho trovato Truvada.",
                    "sources": [{"medicine_id": "00011", "id": "Truvada", "page": 2}],
                },
            ],
            ["00011"],
        )
        chunk = {
            "id": "Farmaco di prova",
            "medicine_id": "00011",
            "page": 1,
            "score": 0.9,
            "text": "Indicazioni del farmaco.",
        }

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(self.service, "_openai_client", return_value=Mock()),
            patch.object(
                self.service, "_condense", return_value="mal di pancia"
            ) as condense,
            patch.object(self.service, "_retrieve") as retrieve_one,
            patch.object(self.service, "_retrieve_all", return_value=[chunk]) as retrieve_all,
            patch.object(
                self.service, "_generate", return_value="Risposta sul nuovo argomento."
            ),
            patch.object(ChatService, "_count_tokens", return_value=10),
        ):
            result = self.service.respond(
                {
                    "session_id": "topic-change-symptom",
                    "message": "Cosa posso prendere per il mal di pancia?",
                }
            )

        self.assertEqual(result["answer"], "Risposta sul nuovo argomento.")
        self.assertEqual(condense.call_args.args[2], [])
        retrieve_one.assert_not_called()
        retrieve_all.assert_called_once()

    def test_legacy_indicazioni_terapeutiche_section_is_supported(self) -> None:
        section = ChatService._indication_section(
            [
                "composizione indicazioni terapeutiche trattamento del mal di testa "
                "controindicazioni non usare se allergico posologia applicare una volta"
            ]
        )

        self.assertIsNotNone(section)
        self.assertIn("mal di testa", section[0])

    def test_indication_section_stops_at_che_cosa_deve_sapere(self) -> None:
        section = ChatService._indication_section(
            [
                "1 che cos'e il farmaco e a cosa serve indicato per la terapia. "
                "2 che cosa deve sapere prima di prendere il farmaco. "
                "Gli antiretrovirali usati per trattare l'HIV possono interagire."
            ]
        )

        self.assertIsNotNone(section)
        self.assertIn("indicato per la terapia", section[0])
        self.assertNotIn("hiv", section[0])

    def test_indication_search_excludes_expired_stock(self) -> None:
        StockRepository(self.database).create(
            {
                "farmaco_id": "00011",
                "data_scadenza": "2000-01-01",
                "nr_pezzi": 1,
            }
        )
        page = SimpleNamespace(
            extract_text=lambda: (
                "1. A cosa serve. Trattamento dell'infezione da HIV. "
                "2. Cosa deve sapere prima di prendere questo medicinale."
            )
        )
        with (
            patch.object(
                self.service.leaflet_store,
                "find",
                side_effect=lambda medicine_id: (
                    Path(self.temp_dir.name) / "00011.pdf"
                    if medicine_id == "00011"
                    else None
                ),
            ),
            patch("pypdf.PdfReader", return_value=SimpleNamespace(pages=[page])),
        ):
            result = self.service._search_stocked_medicines_by_indication(
                ("mal di testa",)
            )

        self.assertEqual(result["matches"], [])

    def test_inventory_tool_returns_expired_and_valid_lots_separately(self) -> None:
        result = self.service._execute_inventory_tool(
            {"medicine_name": "Ximocillina"}
        )

        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"][0]["medicine_id"], "00013")
        self.assertEqual(result["matches"][0]["pezzi_non_scaduti"], 12)
        self.assertEqual(result["matches"][0]["pezzi_scaduti"], 2)

    def test_inventory_tool_without_filter_returns_all_stock_ranked_by_valid_pieces(
        self,
    ) -> None:
        result = self.service._execute_inventory_tool({"medicine_name": ""})

        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"][0]["medicine_id"], "00013")
        self.assertEqual(result["matches"][0]["pezzi_non_scaduti"], 12)
        self.assertEqual(result["matches"][0]["pezzi_scaduti"], 2)

    def test_inventory_tool_treats_medicine_class_as_global_search(self) -> None:
        result = self.service._execute_inventory_tool(
            {"medicine_name": "antibiotici"}
        )

        self.assertEqual(result["query"], "")
        self.assertEqual(
            {match["medicine_id"] for match in result["matches"]}, {"00013"}
        )

    def test_inventory_intent_does_not_capture_dosage_frequency(self) -> None:
        self.assertTrue(ChatService._is_inventory_question("Quante Ximocillina ho?"))
        self.assertTrue(
            ChatService._is_inventory_question("Quali farmaci sono scaduti?")
        )
        self.assertFalse(
            ChatService._is_inventory_question(
                "Quante volte al giorno devo assumere Ximocillina?"
            )
        )
        self.assertTrue(ChatService._is_inventory_question("Ho degli antibiotici?"))
        self.assertFalse(
            ChatService._is_inventory_question(
                "Ho un farmaco a base di paracetamolo?"
            )
        )

    def test_expired_stock_question_lists_all_medicines_without_selection(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(self.service, "_openai_client") as openai_client,
        ):
            result = self.service.respond(
                {
                    "session_id": "expired-session",
                    "message": "Quali farmaci sono scaduti?",
                }
            )

        self.assertEqual(result["medicine_ids"], ["00013"])
        self.assertIn("AMOXICILLINA E ACIDO CLAVULANICO ABC", result["answer"])
        self.assertIn("31/01/2025", result["answer"])
        self.assertNotIn("2027", result["answer"])
        self.assertFalse(result.get("needs_clarification", False))
        openai_client.assert_not_called()

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
        self.assertEqual(
            client.responses.create.call_args_list[0].kwargs["tool_choice"],
            "required",
        )
        second_input = client.responses.create.call_args_list[1].kwargs["input"]
        self.assertEqual(second_input[-1]["type"], "function_call_output")

    def test_inventory_tool_can_be_called_without_a_medicine_name(self) -> None:
        call = SimpleNamespace(
            type="function_call",
            name="get_medicine_stock",
            arguments=json.dumps({"medicine_name": ""}),
            call_id="call-all-stock",
        )
        client = Mock()
        client.responses.create.side_effect = [
            SimpleNamespace(output=[call], output_text=""),
            SimpleNamespace(output=[], output_text="Il farmaco più disponibile è..."),
        ]

        answer, tool_results = self.service._answer_inventory(
            client,
            "gpt-4o-mini",
            [],
            "Quale farmaco ha più scorte non scadute?",
            [],
        )

        self.assertIn("più disponibile", answer)
        self.assertEqual(tool_results[0]["matches"][0]["medicine_id"], "00013")

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

    def test_inventory_follow_up_keeps_only_the_medicine_named_in_the_answer(self) -> None:
        medicines = [
            ("00013", "AMOXICILLINA E ACIDO CLAVULANICO ABC"),
            ("00014", "Farmaco aggiuntivo 1"),
            ("00015", "Farmaco aggiuntivo 2"),
            ("00016", "Farmaco aggiuntivo 3"),
            ("00017", "Farmaco aggiuntivo 4"),
            ("00018", "Farmaco aggiuntivo 5"),
        ]
        repository = MedicineRepository(self.database)
        for medicine_id, name in medicines[1:]:
            repository.create({"id": medicine_id, "nome": name})

        inventory_matches = [
            {
                "medicine_id": medicine_id,
                "name": name,
                "pezzi_non_scaduti": 20 - index,
            }
            for index, (medicine_id, name) in enumerate(medicines)
        ]
        chunk = {
            "id": medicines[0][1],
            "medicine_id": medicines[0][0],
            "page": 2,
            "score": 0.9,
            "text": "Testo del bugiardino.",
        }
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(self.service, "_openai_client", return_value=Mock()),
            patch.object(
                self.service,
                "_answer_inventory",
                return_value=(
                    f"Le maggiori scorte sono di {medicines[0][1]}.",
                    [{"query": "", "matches": inventory_matches}],
                ),
            ),
            patch.object(self.service, "_retrieve", return_value=[chunk]) as retrieve,
            patch.object(
                self.service, "_condense", side_effect=lambda _, __, ___, query, ____: query
            ),
            patch.object(self.service, "_generate", return_value="Risposta dal bugiardino."),
            patch.object(ChatService, "_count_tokens", return_value=10),
        ):
            first = self.service.respond(
                {
                    "session_id": "inventory-follow-up",
                    "message": "Quale farmaco ha più scorte non scadute?",
                }
            )
            saved = self.service.session("inventory-follow-up")
            second = self.service.respond(
                {
                    "session_id": "inventory-follow-up",
                    "message": "E posso prenderlo a stomaco vuoto?",
                }
            )

        self.assertEqual(first["medicine_ids"], ["00013"])
        self.assertEqual(saved["medicine_ids"], ["00013"])
        self.assertEqual(second["medicine_ids"], ["00013"])
        retrieve.assert_called_once()
        self.assertEqual(retrieve.call_args.args[1]["id"], "00013")

    def test_chat_accepts_questions_about_more_than_four_named_medicines(self) -> None:
        medicines = [
            ("00011", "Farmaco di prova"),
            ("00012", "Altro farmaco"),
            ("00013", "AMOXICILLINA E ACIDO CLAVULANICO ABC"),
            ("00014", "Farmaco aggiuntivo 1"),
            ("00015", "Farmaco aggiuntivo 2"),
        ]
        repository = MedicineRepository(self.database)
        repository.create({"id": medicines[3][0], "nome": medicines[3][1]})
        repository.create({"id": medicines[4][0], "nome": medicines[4][1]})
        chunks = [
            {
                "id": name,
                "medicine_id": medicine_id,
                "page": 1,
                "score": 0.9,
                "text": f"Testo {name}",
            }
            for medicine_id, name in medicines
        ]
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(self.service, "_openai_client", return_value=Mock()),
            patch.object(
                self.service, "_retrieve", side_effect=[[chunk] for chunk in chunks]
            ) as retrieve,
            patch.object(
                self.service, "_condense", side_effect=lambda _, __, ___, query, ____: query
            ),
            patch.object(self.service, "_generate", return_value="Confronto documentato."),
            patch.object(ChatService, "_count_tokens", return_value=10),
        ):
            result = self.service.respond(
                {
                    "session_id": "five-medicines",
                    "message": "Confronta " + ", ".join(name for _, name in medicines),
                }
            )

        self.assertEqual(len(result["medicine_ids"]), 5)
        self.assertEqual(retrieve.call_count, 5)

    def test_session_id_cannot_escape_storage_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "ID sessione"):
            self.service.session("../outside")


if __name__ == "__main__":
    unittest.main()