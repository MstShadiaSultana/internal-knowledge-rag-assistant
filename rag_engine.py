"""Small, local retrieval-augmented question-answering prototype."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import faiss
import numpy as np

from document_loader import iter_document_texts


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
FALLBACK_ANSWER = "I could not find enough information in the provided documents."


def _normalized_words(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


@dataclass(frozen=True)
class Chunk:
    source: str
    number: int
    text: str
    page: int | None = None


class RAGEngine:
    def __init__(
        self,
        data_dir: str | Path | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-small",
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        min_similarity: float = 0.35,
        embedder: Any = None,
        generator: Any = None,
    ) -> None:
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")
        if not -1 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between -1 and 1")
        self.data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_similarity = min_similarity
        self.chunks = self._read_documents()

        # Imports and downloads are deferred when test doubles are supplied.
        if embedder is None:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(embedding_model)
        self.embedder = embedder
        self.generator = generator
        self.generator_model = generator_model

        vectors = self._encode([chunk.text for chunk in self.chunks])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(
            self.embedder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype="float32",
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
            raise ValueError("Embedding model returned an unexpected shape")
        return np.ascontiguousarray(vectors)

    def _split_text(self, text: str) -> list[str]:
        text = " ".join(text.split())
        if not text:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                last_space = text.rfind(" ", start + self.chunk_size // 2, end)
                if last_space > start:
                    end = last_space
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end == len(text):
                break
            # Advance at least one character even with long unbroken words.
            start = max(start + 1, end - self.chunk_overlap)
        return chunks

    def _read_documents(self) -> list[Chunk]:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Missing document directory: {self.data_dir}")
        chunks: list[Chunk] = []
        source_counts: dict[str, int] = {}
        for document in iter_document_texts(self.data_dir):
            for piece in self._split_text(document.text):
                number = source_counts.get(document.source, 0) + 1
                source_counts[document.source] = number
                chunks.append(Chunk(document.source, number, piece, document.page))
        if not chunks:
            raise ValueError(f"No extractable .txt, .md or .pdf documents in {self.data_dir}")
        return chunks

    def retrieve(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not question.strip():
            raise ValueError("Question must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = self._encode([question.strip()])
        scores, indices = self.index.search(query, min(top_k, len(self.chunks)))
        return [
            {
                "source": self.chunks[int(index)].source,
                "chunk_id": self.chunks[int(index)].number,
                "page": self.chunks[int(index)].page,
                "chunk": self.chunks[int(index)].text,
                "similarity": round(float(score), 4),
            }
            for score, index in zip(scores[0], indices[0])
        ]

    def answer(self, question: str, top_k: int = 3) -> dict[str, Any]:
        retrieved = self.retrieve(question, top_k)
        if retrieved[0]["similarity"] < self.min_similarity:
            return {"answer": FALLBACK_ANSWER, "evidence": None, "sources": retrieved}
        if self.generator is None:
            from transformers import pipeline

            self.generator = pipeline("text2text-generation", model=self.generator_model)
        context = "\n\n".join(
            f"[{number}] {item['source']} "
            f"(chunk {item['chunk_id']}"
            f"{', page ' + str(item['page']) if item['page'] else ''}): {item['chunk']}"
            for number, item in enumerate(retrieved, start=1)
        )
        prompt = (
            "Give a short answer by copying the relevant phrase or sentence exactly from "
            "the context. Do not use document titles as an answer. "
            f"If no passage answers the question, say: {FALLBACK_ANSWER}\n\n"
            f"Context:\n{context}\n\nQuestion: {question.strip()}\nAnswer:"
        )
        output = self.generator(prompt, max_new_tokens=180, do_sample=False)
        answer = output[0]["generated_text"].strip() or FALLBACK_ANSWER
        normalized_answer = _normalized_words(answer)
        supporting_source = next(
            (
                item for item in retrieved
                if normalized_answer
                and normalized_answer in _normalized_words(item["chunk"])
            ),
            None,
        )
        answer_words = normalized_answer.split()
        question_words = set(_normalized_words(question).split())
        if len(answer_words) <= 2 and all(word in question_words for word in answer_words):
            supporting_source = None
        if supporting_source is None:
            answer = FALLBACK_ANSWER
        # A copied phrase can be checked, but this cannot prove semantic relevance.
        return {"answer": answer, "evidence": supporting_source, "sources": retrieved}


if __name__ == "__main__":
    print(RAGEngine().answer("How should employees use AI-generated information?"))
