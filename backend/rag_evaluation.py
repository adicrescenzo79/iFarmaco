"""Small, repeatable retrieval and grounding checks for the leaflet RAG."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from backend.chat_service import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RETRIEVAL_MODE,
    EMBEDDING_MODELS,
    CHAT_MODELS,
    RETRIEVAL_MODES,
    ChatService,
)
from backend.config import (
    DATABASE_PATH,
    MEDICINES_CSV_PATH,
    STOCK_CSV_PATH,
)
from backend.database import Database
from backend.repositories import MedicineRepository


MEDICINE_ID = "0000038068"


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    question: str
    expected_evidence: str | None
    expected_page: int | None = None


CASES = (
    EvaluationCase(
        "dose-raccomandata",
        "Qual è la dose raccomandata per adulti e bambini di almeno 40 kg?",
        "1 bustina due volte al giorno",
        3,
    ),
    EvaluationCase(
        "dose-piu-alta",
        "Qual è la dose più alta indicata per gli adulti?",
        "1 bustina tre volte al giorno",
        3,
    ),
    EvaluationCase(
        "durata",
        "Per quanto tempo si può prendere questo medicinale?",
        "per più di 2 settimane",
        3,
    ),
    EvaluationCase(
        "fuori-corpus",
        "Qual è il prezzo al pubblico di questo medicinale?",
        None,
    ),
)


def evidence_rank(
    chunks: list[dict[str, Any]],
    evidence: str,
    expected_page: int | None = None,
) -> int | None:
    """Return the one-based rank of the first chunk containing expected evidence."""
    needle = " ".join(evidence.casefold().split())
    for rank, chunk in enumerate(chunks, start=1):
        if expected_page is not None and chunk.get("page") != expected_page:
            continue
        text = " ".join(str(chunk.get("text", "")).casefold().split())
        if needle in text:
            return rank
    return None


def context_tokens(chunks: list[dict[str, Any]]) -> int:
    documents = "\n\n".join(
        f"<documento farmaco=\"{chunk['id']}\" pagina=\"{chunk['page']}\">\n"
        f"{chunk['text']}\n</documento>"
        for chunk in chunks
    )
    return ChatService._count_tokens([{"role": "user", "content": documents}])


def move_evidence_chunk(
    chunks: list[dict[str, Any]], evidence_rank_value: int, position: int
) -> list[dict[str, Any]]:
    """Reorder a fixed set of chunks, moving the evidence chunk to a zero-based slot."""
    if not 1 <= evidence_rank_value <= len(chunks):
        raise ValueError("Il rango del passaggio atteso non è valido.")
    if not 0 <= position < len(chunks):
        raise ValueError("La posizione richiesta non è valida.")
    target = chunks[evidence_rank_value - 1]
    others = [
        chunk for index, chunk in enumerate(chunks) if index != evidence_rank_value - 1
    ]
    return [*others[:position], target, *others[position:]]


def _print_retrieval_result(
    case: EvaluationCase, chunks: list[dict[str, Any]], top_k: int
) -> int | None:
    print(f"\n[{case.name}] {case.question}")
    if case.expected_evidence is None:
        print("  Risposta attesa fuori dal corpus: da verificare manualmente.")
        return None

    rank = evidence_rank(chunks, case.expected_evidence, case.expected_page)
    if rank is None:
        print(f"  MISS: passaggio atteso assente nei {len(chunks)} risultati.")
    elif rank <= top_k:
        print(f"  HIT@{top_k}: passaggio atteso al rango {rank}.")
    else:
        print(f"  MISS@{top_k}: passaggio atteso al rango {rank}.")
    for index, chunk in enumerate(chunks, start=1):
        page = chunk.get("page", "?")
        print(f"  {index}. pagina {page}, score {chunk.get('score', 0):.3f}")
    return rank


def _print_generation(
    service: ChatService,
    client: Any,
    model: str,
    case: EvaluationCase,
    chunks: list[dict[str, Any]],
) -> None:
    usage: list[dict[str, Any]] = []
    answer = service._generate(
        model, chunks, case.question, [], usage_records=usage
    )
    print(f"  Risposta: {answer}")
    if case.expected_evidence is None:
        print("  Controllo manuale: verifica che non inventi il prezzo.")
    else:
        citation = f"p. {case.expected_page}"
        print(
            "  Controllo manuale: verifica il fatto atteso "
            f"«{case.expected_evidence}» e la citazione a {citation}."
        )
    if usage and "input_tokens" in usage[0]:
        print(
            f"  Token generazione: {usage[0]['input_tokens']} in, "
            f"{usage[0]['output_tokens']} out."
        )


def run_evaluation(
    *,
    top_k: int,
    embedding_model: str,
    retrieval_mode: str,
    model: str,
    generate: bool,
) -> int:
    if not ChatService.configuration()["configured"]:
        raise RuntimeError("Configura OPENAI_API_KEY nel file .env.")

    database = Database(
        DATABASE_PATH,
        medicines_csv=MEDICINES_CSV_PATH,
        stock_csv=STOCK_CSV_PATH,
    )
    database.initialize()
    service = ChatService(database)
    medicine = MedicineRepository(service.database).get(MEDICINE_ID)
    client = service._openai_client()
    results: dict[str, list[dict[str, Any]]] = {}
    hit_ranks: list[int] = []
    misses = 0
    candidate_k = min(8, max(top_k, top_k * 2))

    print(
        f"Farmaco: {medicine['nome']} ({MEDICINE_ID}) | k={top_k} | "
        f"candidati={candidate_k} | retrieval={retrieval_mode} | "
        f"embedding={embedding_model}"
    )
    for case in CASES:
        chunks = service._retrieve(
            client,
            medicine,
            case.question,
            candidate_k,
            embedding_model,
            search_mode=retrieval_mode,
        )
        results[case.name] = chunks
        rank = _print_retrieval_result(case, chunks, top_k)
        if case.expected_evidence is not None:
            if rank is None or rank > top_k:
                misses += 1
            else:
                hit_ranks.append(rank)
        print(
            f"  Token del contesto passato al modello (top-{top_k}): "
            f"{context_tokens(chunks[:top_k])}"
        )

    positive_cases = [case for case in CASES if case.expected_evidence is not None]
    reciprocal_rank = (
        sum(1 / rank for rank in hit_ranks) / len(positive_cases)
        if positive_cases
        else 0.0
    )
    recall = len(hit_ranks) / len(positive_cases) if positive_cases else 0.0
    print(
        f"\nRecall@{top_k}: {len(hit_ranks)}/{len(positive_cases)} "
        f"({recall:.0%}) | MRR@{top_k}: {reciprocal_rank:.3f}"
    )
    print(
        "Il budget memoria limita la cronologia, non questi chunk: "
        "confronta separatamente i token del contesto."
    )

    if not generate:
        print(
            "\nNessuna chiamata al modello di risposta. "
            "Aggiungi --generate per controllare grounding e lost-in-the-middle."
        )
        return 1 if misses else 0

    for case in CASES:
        print(f"\nGenerazione [{case.name}]")
        _print_generation(
            service, client, model, case, results[case.name][:top_k]
        )

    case = CASES[0]
    chunks = results[case.name][:top_k]
    rank = evidence_rank(
        chunks, case.expected_evidence or "", case.expected_page
    )
    if rank is None or len(chunks) < 2:
        print(
            "\nLost-in-the-middle: non valutabile perché il passaggio atteso "
            "non è stato recuperato insieme ad altri chunk."
        )
    else:
        target_positions = [(0, "inizio")]
        if len(chunks) > 2:
            target_positions.append((len(chunks) // 2, "centro"))
        target_positions.append((len(chunks) - 1, "fine"))
        for position, label in target_positions:
            ordered = move_evidence_chunk(chunks, rank, position)
            print(f"\nLost-in-the-middle [{label}]")
            print(f"  Token del contesto: {context_tokens(ordered)}")
            _print_generation(service, client, model, case, ordered)

    return 1 if misses else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valuta retrieval e grounding sui bugiardini di iFarmaco."
    )
    parser.add_argument("--top-k", type=int, default=4, choices=range(1, 9))
    parser.add_argument(
        "--embedding-model",
        choices=EMBEDDING_MODELS,
        default=DEFAULT_EMBEDDING_MODEL,
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=RETRIEVAL_MODES,
        default=DEFAULT_RETRIEVAL_MODE,
    )
    parser.add_argument("--model", choices=CHAT_MODELS, default=CHAT_MODELS[0])
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Chiama il modello di risposta per grounding e lost-in-the-middle.",
    )
    args = parser.parse_args()
    return run_evaluation(
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        retrieval_mode=args.retrieval_mode,
        model=args.model,
        generate=args.generate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
