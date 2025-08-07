import streamlit as st
from utils.build_graph import retrieve_from_graph
from utils.enhanced_processing import (
    QueryProcessor,
    AdvancedRetriever,
    ResultPostProcessor,
)
from langchain_core.documents import Document
import requests
import numpy as np


# Initialize enhanced components
query_processor = QueryProcessor()
advanced_retriever = AdvancedRetriever()
result_processor = ResultPostProcessor()


# Query Expansion with HyDE
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


# 🚀 Enhanced Retrieval Pipeline with Smart Context Detection
def retrieve_documents(query, uri, model, chat_history=""):
    """Enhanced retrieval pipeline with dynamic context detection and improved ranking"""

    # Step 1: Pre-processing - Detect query intent and context
    st.write("🔍 **Step 1: Query Analysis**")
    intent = query_processor.detect_intent(query)
    contexts = query_processor.extract_context_keywords(query)

    # Dynamic context adjustment based on intent
    base_max_contexts = st.session_state.max_contexts
    if intent in ["list", "count"]:
        # For list/count queries, we need more contexts to be comprehensive
        dynamic_max_contexts = max(
            base_max_contexts * 2, 15
        )  # At least 15 for count/list
        st.write(
            f"🤖 **Smart Context Boost**: Increased from {base_max_contexts} to {dynamic_max_contexts} for {intent.upper()} query"
        )
    elif intent == "summary":
        # For summaries, we also need more contexts to cover all aspects
        dynamic_max_contexts = max(base_max_contexts * 1.5, 10)
        st.write(
            f"🔄 **Smart Context Boost**: Increased from {base_max_contexts} to {dynamic_max_contexts} for {intent.upper()} query"
        )
    else:
        dynamic_max_contexts = base_max_contexts

    st.write(f"🏷️ Detected Intent: **{intent.upper()}**")
    st.write(f"🏷️ Detected Contexts: **{', '.join(contexts)}**")
    st.write(f"📊 Target Contexts: **{dynamic_max_contexts}**")

    # Step 2: Context-based chunk search
    st.write("\n🔍 **Step 2: Context-Based Search**")
    chunks = st.session_state.retrieval_pipeline.get("chunks", [])

    # Search chunks using detected contexts
    context_relevant_chunks = advanced_retriever.search_chunks_by_context(
        chunks, contexts, query, top_k=20  # Get more candidates for filtering
    )

    st.write(f"📝 Context-Relevant Chunks Found: **{len(context_relevant_chunks)}**")
    for i, chunk in enumerate(context_relevant_chunks[:3]):  # Show first 3
        st.write(f"Context Chunk {i+1}: {chunk.page_content[:150]}...")

    # Step 3: Traditional hybrid retrieval with expanded query
    st.write("\n🔍 **Step 3: Hybrid Retrieval (BM25+FAISS)**")
    expanded_query = (
        expand_query(f"{chat_history}\n{query}", uri, model)
        if st.session_state.enable_hyde
        else query_processor.expand_query_with_context(query, contexts)
    )

    # Get hybrid results
    hybrid_docs = st.session_state.retrieval_pipeline["ensemble"].invoke(expanded_query)
    st.write(f"🔍 Hybrid Retrieved: **{len(hybrid_docs)} chunks**")
    for i, doc in enumerate(hybrid_docs[:3]):  # Show first 3
        st.write(f"Hybrid Chunk {i+1}: {doc.page_content[:150]}...")

    # Step 4: GraphRAG Retrieval with Enhanced Processing
    graph_chunks = []
    if st.session_state.enable_graph_rag:
        st.write("\n🔍 **Step 4: GraphRAG Enhanced Retrieval**")
        graph_results = retrieve_from_graph(
            query, st.session_state.retrieval_pipeline["knowledge_graph"]
        )

        if graph_results:
            st.write(f"GraphRAG Retrieved Nodes: **{graph_results}**")

            # Enhanced GraphRAG chunk retrieval
            graph_chunks = advanced_retriever.enhance_graphrag_results(
                graph_results, chunks, query
            )

            if graph_chunks:
                st.write(f"🔗 GraphRAG Enhanced Chunks: **{len(graph_chunks)}**")
                for i, chunk in enumerate(graph_chunks[:3]):  # Show first 3
                    st.write(f"GraphRAG Chunk {i+1}: {chunk.page_content[:150]}...")

    # Step 5: Combine results intelligently
    st.write("\n🔍 **Step 5: Result Combination**")

    # Combine context-relevant, hybrid, and graph results
    all_docs = context_relevant_chunks + hybrid_docs
    combined_docs = result_processor.combine_results(
        all_docs, graph_chunks, dynamic_max_contexts + 5  # Get extra for re-ranking
    )

    st.write(f"🔄 Combined Unique Chunks: **{len(combined_docs)}**")

    # Step 6: Enhanced Re-ranking
    final_docs = combined_docs
    if st.session_state.enable_reranking and st.session_state.retrieval_pipeline.get(
        "reranker"
    ):
        st.write("\n🔍 **Step 6: Enhanced Re-ranking**")

        # Use improved re-ranking with NaN handling
        final_docs = advanced_retriever.improved_reranking(
            query, combined_docs, st.session_state.retrieval_pipeline["reranker"]
        )

        st.write("🤖 **Re-ranked Results (Top to Bottom):**")
        for i, doc in enumerate(final_docs[:5]):  # Show top 5
            st.write(f"Re-ranked #{i+1}: {doc.page_content[:150]}...")

    # Return final results limited by dynamic max_contexts
    final_results = final_docs[:dynamic_max_contexts]

    st.write(
        f"\n✅ **Final Results**: **{len(final_results)} chunks** selected for context"
    )

    return final_results
