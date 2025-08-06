import streamlit as st
from utils.build_graph import retrieve_from_graph
from langchain_core.documents import Document
import requests


# 🚀 Query Expansion with HyDE
def expand_query(query, uri, model):
    try:
        response = requests.post(
            uri,
            json={
                "model": model,
                "prompt": f"Generate a hypothetical answer to: {query}",
                "stream": False,
            },
        ).json()
        return f"{query}\n{response.get('response', '')}"
    except Exception as e:
        st.error(f"Query expansion failed: {str(e)}")
        return query


# 🚀 Advanced Retrieval Pipeline
def retrieve_documents(query, uri, model, chat_history=""):
    # Debug: Show all chunks containing 'case study' (case-insensitive)
    all_case_study_chunks = [
        chunk
        for chunk in st.session_state.retrieval_pipeline.get("chunks", [])
        if "case study" in chunk.page_content.lower()
    ]
    st.write(f"📝 All Chunks with 'case study': {len(all_case_study_chunks)} found")
    for i, chunk in enumerate(all_case_study_chunks):
        st.write(f"Case Study Chunk {i+1}: {chunk.page_content[:200]}...")
    expanded_query = (
        expand_query(f"{chat_history}\n{query}", uri, model)
        if st.session_state.enable_hyde
        else query
    )

    # 🔍 Retrieve documents using BM25 + FAISS
    docs = st.session_state.retrieval_pipeline["ensemble"].invoke(expanded_query)
    st.write("\n---\n<b>🔍 Hybrid (BM25+FAISS) Top Chunks:</b>", unsafe_allow_html=True)
    for i, doc in enumerate(docs):
        st.write(f"Hybrid Chunk {i+1}: {doc.page_content[:200]}...")

    # 🚀 GraphRAG Retrieval
    if st.session_state.enable_graph_rag:
        graph_results = retrieve_from_graph(
            query, st.session_state.retrieval_pipeline["knowledge_graph"]
        )
        st.write(f"🔍 GraphRAG Retrieved Nodes: {graph_results}")

        # Return the full text chunk associated with each matched node (not just the node label)
        # Find all chunks that contain the matched node/entity
        graph_chunks = []
        chunks = st.session_state.retrieval_pipeline.get("chunks", [])
        for node in graph_results:
            for chunk in chunks:
                if node in chunk.page_content and chunk not in graph_chunks:
                    graph_chunks.append(chunk)
        if graph_chunks:
            st.write("\n---\n<b>🔗 GraphRAG Top Chunks:</b>", unsafe_allow_html=True)
            for i, chunk in enumerate(graph_chunks):
                st.write(f"GraphRAG Chunk {i+1}: {chunk.page_content[:200]}...")
            docs = (
                graph_chunks + docs
            )  # Merge GraphRAG results with FAISS + BM25 results

    # 🚀 Neural Re-ranking (if enabled)
    if st.session_state.enable_reranking:
        pairs = [
            [query, doc.page_content] for doc in docs
        ]  # ✅ Fix: Use `page_content`
        scores = st.session_state.retrieval_pipeline["reranker"].predict(pairs)

        # Sort documents based on reranking scores
        ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
        st.write(
            "\n---\n<b>🤖 Re-ranked Chunks (Top to Bottom):</b>", unsafe_allow_html=True
        )
        for i, (doc, score) in enumerate(
            sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        ):
            st.write(
                f"Re-ranked Chunk {i+1} (score={score:.4f}): {doc.page_content[:200]}..."
            )
    else:
        ranked_docs = docs

    return ranked_docs[
        : st.session_state.max_contexts
    ]  # Return top results based on max_contexts
