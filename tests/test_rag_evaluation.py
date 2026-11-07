import unittest

from backend.rag_evaluation import (
    CASES,
    context_tokens,
    evidence_rank,
    move_evidence_chunk,
)


class RagEvaluationTests(unittest.TestCase):
    def test_evidence_rank_is_one_based_and_case_insensitive(self) -> None:
        chunks = [
            {"text": "Testo irrilevante"},
            {"text": "Dose raccomandata: 1 bustina due volte al giorno."},
        ]

        self.assertEqual(
            evidence_rank(chunks, "1 bustina due volte al giorno"), 2
        )

    def test_evidence_rank_returns_none_when_phrase_is_missing(self) -> None:
        self.assertIsNone(evidence_rank([{"text": "Testo disponibile"}], "Prezzo"))

    def test_evidence_rank_can_require_the_expected_page(self) -> None:
        chunks = [
            {"page": 2, "text": "Dose raccomandata: 1 bustina al giorno."},
            {"page": 3, "text": "Dose raccomandata: 1 bustina al giorno."},
        ]

        self.assertEqual(
            evidence_rank(chunks, "1 bustina al giorno", expected_page=3), 2
        )

    def test_reordering_keeps_same_chunks_and_moves_evidence(self) -> None:
        chunks = [
            {"id": "a", "text": "Distrattore"},
            {"id": "b", "text": "Passaggio atteso"},
            {"id": "c", "text": "Altro distrattore"},
        ]

        moved = move_evidence_chunk(chunks, 2, 2)

        self.assertEqual([chunk["id"] for chunk in moved], ["a", "c", "b"])
        self.assertCountEqual(
            [chunk["id"] for chunk in moved], [chunk["id"] for chunk in chunks]
        )

    def test_context_token_count_is_positive(self) -> None:
        self.assertGreater(
            context_tokens([{"id": "Farmaco", "page": 1, "text": "Testo."}]),
            0,
        )

    def test_benchmark_has_answerable_and_unanswerable_cases(self) -> None:
        self.assertTrue(any(case.expected_evidence for case in CASES))
        self.assertTrue(any(case.expected_evidence is None for case in CASES))


if __name__ == "__main__":
    unittest.main()
