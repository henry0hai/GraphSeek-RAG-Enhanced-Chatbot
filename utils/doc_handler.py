import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from utils.build_graph import build_knowledge_graph
from utils.adaptive_chunking import AdaptiveChunker
from rank_bm25 import BM25Okapi
import os
import re


def process_documents(uploaded_files, reranker, embedding_model, base_url):
    if st.session_state.documents_loaded:
        return

    st.session_state.processing = True
    documents = []

    # Create temp directory
    if not os.path.exists("temp"):
        os.makedirs("temp")

    # Process files
    for file in uploaded_files:
        try:
            file_path = os.path.join("temp", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                continue

            documents.extend(loader.load())
            os.remove(file_path)
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return

    # 🚀 Advanced Adaptive Text Splitting
    # Initialize adaptive chunker for intelligent document processing
    adaptive_chunker = AdaptiveChunker()
    
    # Let user choose chunking strategy
    st.write("⚙️ **Chunking Strategy Selection**")
    chunking_strategy = st.selectbox(
        "Choose optimal chunking strategy for your use case:",
        options=["balanced", "detailed", "contextual", "count_optimized"],
        index=0,
        help="Different strategies optimize for different query types"
    )
    
    # Get optimal splitter based on document characteristics and user preference
    text_splitter = adaptive_chunker.get_optimal_chunker(documents, chunking_strategy)
    
    # Split documents with quality analysis
    try:
        texts = text_splitter.split_documents(documents)
        
        # Analyze chunk quality
        quality_analysis = adaptive_chunker.analyze_chunk_quality(texts)
        
        # Display analysis results
        if quality_analysis:
            st.write("📊 **Chunk Quality Analysis**:")
            st.write(f"   - Total chunks: {quality_analysis['total_chunks']}")
            st.write(f"   - Average length: {quality_analysis['avg_length']:.0f} characters")
            st.write(f"   - Size range: {quality_analysis['min_length']}-{quality_analysis['max_length']} chars")
            st.write(f"   - Quality score: {quality_analysis['quality_score']:.0f}/100")
            
            # Show recommendations
            if quality_analysis['recommendations']:
                st.write("🚀 **Recommendations**:")
                for rec in quality_analysis['recommendations']:
                    st.write(f"   - {rec}")
        
    except Exception as e:
        st.error(f"Chunking failed: {e}")
        # Fallback to simple splitter
        st.write("🔄 Using fallback chunking strategy...")
        fallback_splitter = CharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150,
            separator="\n",
        )
        texts = fallback_splitter.split_documents(documents)
    text_contents = [doc.page_content for doc in texts]

    # 🚀 Hybrid Retrieval Setup
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)

    # Vector store
    vector_store = FAISS.from_documents(texts, embeddings)

    # BM25 store
    bm25_retriever = BM25Retriever.from_texts(
        text_contents,
        bm25_impl=BM25Okapi,
        preprocess_func=lambda text: re.sub(r"\W+", " ", text).lower().split(),
    )

    # Ensemble retrieval
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_store.as_retriever(search_kwargs={"k": 5})],
        weights=[0.4, 0.6],
    )

    # Store in session
    st.session_state.retrieval_pipeline = {
        "ensemble": ensemble_retriever,
        "reranker": reranker,  # Now using the global reranker variable
        "texts": text_contents,
        "chunks": texts,  # Store the full chunk objects for later retrieval
        "knowledge_graph": build_knowledge_graph(texts),  # Store Knowledge Graph
    }

    st.session_state.documents_loaded = True
    st.session_state.processing = False

    # Debugging: Print Knowledge Graph Nodes & Edges
    if "knowledge_graph" in st.session_state.retrieval_pipeline:
        G = st.session_state.retrieval_pipeline["knowledge_graph"]
        st.write(f"🔗 Total Nodes: {len(G.nodes)}")
        st.write(f"🔗 Total Edges: {len(G.edges)}")
        st.write(f"🔗 Sample Nodes: {list(G.nodes)[:10]}")
        st.write(f"🔗 Sample Edges: {list(G.edges)[:10]}")
