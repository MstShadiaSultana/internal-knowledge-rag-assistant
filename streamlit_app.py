import requests
import streamlit as st

API_URL="http://127.0.0.1:8000/ask"

st.set_page_config(page_title="Internal Knowledge AI Assistant", page_icon="🤖")
st.title("Internal Knowledge AI Assistant")
st.caption("Local RAG demo with Sentence Transformers, FAISS, FastAPI and FLAN-T5")

question=st.text_area(
    "Ask a question about the documents",
    placeholder="Example: How should employees use AI-generated information?"
)
top_k=st.slider("Number of retrieved chunks",1,5,3)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            r=requests.post(API_URL, json={"question":question,"top_k":top_k}, timeout=120)
            if r.status_code != 200:
                st.error(r.text)
            else:
                data=r.json()
                st.subheader("Answer")
                st.write(data["answer"])
                if data.get("evidence"):
                    item = data["evidence"]
                    page = f", page {item['page']}" if item.get("page") else ""
                    st.caption(f"Copied phrase found in: {item['source']} (chunk {item['chunk_id']}{page})")
                st.subheader("Retrieved Passages")
                st.caption("Inspect the passages before relying on the answer. Similarity is not a confidence score.")
                for i,s in enumerate(data["sources"],1):
                    location = f", page {s['page']}" if s.get("page") else ""
                    with st.expander(f"{i}. {s['source']} (chunk {s['chunk_id']}{location}) | similarity={s['similarity']}"):
                        st.write(s["chunk"])
        except requests.exceptions.ConnectionError:
            st.error("Start the API first with: uvicorn api:app --reload")
        except Exception as e:
            st.error(str(e))
