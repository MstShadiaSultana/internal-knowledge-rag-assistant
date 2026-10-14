"""Fast offline behavior tests; no model download is required."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import app, get_engine
from document_loader import iter_document_texts
from rag_engine import RAGEngine


class FakeEmbedder:
    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            text = text.lower()
            if "holiday" in text or "leave" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.array(vectors, dtype="float32")


def fake_generator(prompt, **_kwargs):
    assert "Context:" in prompt and "Question:" in prompt
    answer = (
        "Employees may request holiday leave."
        if "holiday leave" in prompt.split("Question:")[-1]
        else "AI output must be checked by a human."
    )
    return [{"generated_text": answer}]


@pytest.fixture
def engine(tmp_path: Path):
    (tmp_path / "leave.txt").write_text("Employees may request holiday leave.", encoding="utf-8")
    (tmp_path / "ai.txt").write_text("AI output must be checked by a human.", encoding="utf-8")
    return RAGEngine(tmp_path, embedder=FakeEmbedder(), generator=fake_generator)


def test_retrieval_ranks_relevant_source_first(engine):
    results = engine.retrieve("How do I request holiday leave?", top_k=1)
    assert len(results) == 1
    assert results[0]["source"] == "leave.txt"
    assert results[0]["chunk_id"] == 1


def test_answer_returns_inspectable_evidence(engine):
    result = engine.answer("Should I check AI output?", top_k=1)
    assert result["answer"] == "AI output must be checked by a human."
    assert result["evidence"]["source"] == "ai.txt"
    assert result["sources"][0]["source"] == "ai.txt"
    assert "checked by a human" in result["sources"][0]["chunk"]


def test_weak_retrieval_abstains_without_invoking_generator(tmp_path):
    (tmp_path / "leave.txt").write_text("Holiday leave is available.", encoding="utf-8")
    called = []
    def generator(_prompt, **_kwargs):
        called.append(True)
        return [{"generated_text": "unsupported"}]
    rag = RAGEngine(tmp_path, min_similarity=1.0,
                    embedder=FakeEmbedder(), generator=generator)
    # The fake embedding score is 0 for an unrelated question.
    result = rag.answer("What about AI?")
    assert result["answer"] == "I could not find enough information in the provided documents."
    assert result["evidence"] is None
    assert not called


def test_generated_text_not_present_in_evidence_is_rejected(tmp_path):
    (tmp_path / "notes.txt").write_text("Employees may request holiday leave.", encoding="utf-8")
    rag = RAGEngine(
        tmp_path,
        embedder=FakeEmbedder(),
        generator=lambda *_args, **_kwargs: [{"generated_text": "A made-up policy."}],
    )
    assert rag.answer("holiday leave")["answer"].startswith("I could not find")
    assert rag.answer("holiday leave")["evidence"] is None


def test_answer_that_only_repeats_question_is_rejected(tmp_path):
    (tmp_path / "notes.txt").write_text("Never share a password.", encoding="utf-8")
    rag = RAGEngine(
        tmp_path,
        embedder=FakeEmbedder(),
        generator=lambda *_args, **_kwargs: [{"generated_text": "password"}],
    )
    assert rag.answer("What is the password?")["evidence"] is None


def test_empty_documents_fail_clearly(tmp_path):
    (tmp_path / "empty.txt").write_text(" \n ", encoding="utf-8")
    with pytest.raises(ValueError, match="No extractable"):
        RAGEngine(tmp_path, embedder=FakeEmbedder(), generator=fake_generator)


def test_markdown_and_pdf_keep_document_provenance(tmp_path):
    (tmp_path / "notes.md").write_text("# Holiday leave\nAsk your manager.", encoding="utf-8")
    sample_pdf = Path(__file__).resolve().parents[1] / "data" / "sample_incident_response.pdf"
    (tmp_path / sample_pdf.name).write_bytes(sample_pdf.read_bytes())
    texts = list(iter_document_texts(tmp_path))
    assert any(item.source == "notes.md" and item.page is None for item in texts)
    assert any(
        item.source == "sample_incident_response.pdf" and item.page == 2
        and "Do not delete logs" in item.text
        for item in texts
    )


def test_invalid_questions_and_chunk_settings(engine, tmp_path):
    with pytest.raises(ValueError, match="blank"):
        engine.retrieve("  ")
    with pytest.raises(ValueError, match="top_k"):
        engine.retrieve("hello", top_k=0)
    with pytest.raises(ValueError, match="overlap"):
        RAGEngine(tmp_path, chunk_size=20, chunk_overlap=20,
                  embedder=FakeEmbedder(), generator=fake_generator)


def test_chunker_advances_with_long_words(engine):
    engine.chunk_size = 10
    engine.chunk_overlap = 8
    chunks = engine._split_text("abcdefghijklmnopqrstuvwxyz")
    assert len(chunks) < 30
    assert chunks[0] == "abcdefghij"
    assert chunks[-1].endswith("z")


def test_api_validation_and_response(engine):
    app.dependency_overrides[get_engine] = lambda: engine
    try:
        client = TestClient(app)
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/ask", json={"question": " "}).status_code == 422
        assert client.post("/ask", json={"question": "holiday leave", "top_k": 0}).status_code == 422
        response = client.post("/ask", json={"question": "holiday leave", "top_k": 1})
        assert response.status_code == 200
        assert response.json()["sources"][0]["source"] == "leave.txt"
        assert response.json()["evidence"]["source"] == "leave.txt"
    finally:
        app.dependency_overrides.clear()
