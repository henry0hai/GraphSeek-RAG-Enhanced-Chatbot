import streamlit as st
import networkx as nx
import re

try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception as e:
    SPACY_AVAILABLE = False
    nlp = None
    st.warning(
        "spaCy or its model is not installed. Falling back to regex-based entity extraction. For best results, run: pip install spacy && python -m spacy download en_core_web_sm"
    )

# Semantic similarity for node matching
try:
    from sentence_transformers import SentenceTransformer, util

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    ST_AVAILABLE = True
except Exception as e:
    ST_AVAILABLE = False
    st_model = None
    st.warning(
        "sentence-transformers not installed. For best results, run: pip install sentence-transformers"
    )


def build_knowledge_graph(docs):
    G = nx.Graph()
    for doc in docs:
        if SPACY_AVAILABLE and nlp is not None:
            spacy_doc = nlp(doc.page_content)
            # Use a broad set of entity types for generality
            entities = [
                ent.text
                for ent in spacy_doc.ents
                if ent.label_
                in {
                    "PERSON",
                    "ORG",
                    "GPE",
                    "PRODUCT",
                    "EVENT",
                    "WORK_OF_ART",
                    "LAW",
                    "LANGUAGE",
                    "LOC",
                    "FAC",
                    "NORP",
                    "DATE",
                    "TIME",
                }
            ]
        else:
            # Fallback: regex for capitalized phrases
            entities = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b", doc.page_content)
        # Connect all pairs of entities in the same chunk for richer relationships
        if len(entities) > 1:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    G.add_edge(entities[i], entities[j])
    return G


def retrieve_from_graph(query, G, top_k=5):
    st.write(f"🔎 Searching GraphRAG for: {query}")

    node_texts = list(G.nodes)
    matched_nodes = []
    if ST_AVAILABLE and st_model is not None and node_texts:
        # Semantic similarity search
        try:
            node_embeddings = st_model.encode(node_texts, convert_to_tensor=True)
            query_embedding = st_model.encode([query], convert_to_tensor=True)
            import torch

            similarities = util.pytorch_cos_sim(query_embedding, node_embeddings)[0]
            top_indices = similarities.topk(top_k).indices.tolist()
            matched_nodes = [node_texts[i] for i in top_indices]
            st.write(f"🟢 GraphRAG (semantic) Matched Nodes: {matched_nodes}")
        except Exception as e:
            st.warning(
                f"Semantic search failed: {str(e)}. Falling back to substring matching."
            )
    if not matched_nodes:
        # Fallback: substring matching
        query_words = query.lower().split()
        matched_nodes = [
            node
            for node in node_texts
            if any(word in node.lower() for word in query_words)
        ]
        if matched_nodes:
            st.write(f"🟢 GraphRAG (substring) Matched Nodes: {matched_nodes}")

    if matched_nodes:
        related_nodes = []
        for node in matched_nodes:
            related_nodes.extend(list(G.neighbors(node)))  # Get connected nodes
        st.write(f"🟢 GraphRAG Retrieved Related Nodes: {related_nodes[:top_k]}")
        return related_nodes[:top_k]

    st.write(f"❌ No graph results found for: {query}")
    return []
