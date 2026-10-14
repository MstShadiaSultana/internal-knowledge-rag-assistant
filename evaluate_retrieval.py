"""Report retrieval and abstention metrics on the bundled synthetic questions."""

import json
from pathlib import Path

from rag_engine import FALLBACK_ANSWER, RAGEngine


CASES_PATH = Path(__file__).resolve().parent / "evaluation" / "questions.json"


def evaluate(engine: RAGEngine, cases: dict) -> dict:
    positive = cases["positive"]
    ranks = []
    source_top1 = 0
    positive_with_evidence = 0
    for case in positive:
        results = engine.retrieve(case["question"], top_k=3)
        if results[0]["source"] == case["source"]:
            source_top1 += 1
        rank = next(
            (
                position
                for position, result in enumerate(results, start=1)
                if result["source"] == case["source"]
                and ("page" not in case or result["page"] == case["page"])
            ),
            None,
        )
        ranks.append(rank)
        if rank != 1:
            print(
                f"MISS | expected {case['source']} page {case.get('page')} "
                f"| got {results[0]['source']} page {results[0]['page']} "
                f"| {case['question']}"
            )
        answer_result = engine.answer(case["question"], top_k=3)
        if answer_result["evidence"] is not None:
            positive_with_evidence += 1
        else:
            print(f"NO SUPPORTED ANSWER | {case['question']}")

    negative = cases["out_of_scope"]
    retrieval_abstained = 0
    final_abstained = 0
    for question in negative:
        score = engine.retrieve(question, top_k=1)[0]["similarity"]
        retrieval_abstained += score < engine.min_similarity
        final_answer = engine.answer(question, top_k=3)["answer"]
        final_abstained += final_answer == FALLBACK_ANSWER
        if final_answer != FALLBACK_ANSWER:
            print(f"UNSUPPORTED ANSWER | score {score} | {question} | {final_answer}")
    n = len(positive)
    return {
        "positive_questions": n,
        "source_top1": round(source_top1 / n, 3),
        "evidence_top1": round(sum(rank == 1 for rank in ranks) / n, 3),
        "evidence_recall_at_3": round(sum(rank is not None for rank in ranks) / n, 3),
        "evidence_mrr_at_3": round(sum(1 / rank for rank in ranks if rank) / n, 3),
        "positive_with_copied_evidence": positive_with_evidence,
        "out_of_scope_questions": len(negative),
        "out_of_scope_retrieval_abstained": retrieval_abstained,
        "out_of_scope_final_abstained": final_abstained,
        "threshold": engine.min_similarity,
    }


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    engine = RAGEngine()
    print(json.dumps(evaluate(engine, cases), indent=2))
    print("Synthetic demonstration only; no claim-level answer accuracy is measured.")


if __name__ == "__main__":
    main()
