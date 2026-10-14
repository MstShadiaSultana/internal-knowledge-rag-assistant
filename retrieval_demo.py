from rag_engine import RAGEngine

rag=RAGEngine()
question="How should employees use AI-generated information?"
results=rag.retrieve(question, top_k=3)

print("\nQUESTION")
print(question)
print("\nTOP RETRIEVED CHUNKS")
for i,r in enumerate(results,1):
    print(f"\n{i}. Source: {r['source']}")
    print(f"Similarity: {r['similarity']}")
    print(r["chunk"])
