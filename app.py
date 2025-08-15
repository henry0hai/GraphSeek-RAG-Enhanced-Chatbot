import streamlit as st
import requests
import json
from utils.retriever_pipeline import retrieve_documents
from utils.doc_handler import process_documents
from utils.enhanced_processing import QueryProcessor, ResultPostProcessor
from sentence_transformers import CrossEncoder
import torch
import os
from dotenv import load_dotenv, find_dotenv

torch.classes.__path__ = [
    os.path.join(torch.__path__[0], torch.classes.__file__)
]  # Fix for torch classes not found error
load_dotenv(
    find_dotenv()
)  # Loads .env file contents into the application based on key-value pairs defined therein, making them accessible via 'os' module functions like os.getenv().

OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL = os.getenv(
    "MODEL", "deepseek-r1:7b"
)  # Make sure you have it installed in ollama
EMBEDDINGS_MODEL = "nomic-embed-text:latest"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize enhanced processors
query_processor = QueryProcessor()
result_processor = ResultPostProcessor()

reranker = None  # 🚀 Initialize Cross-Encoder (Reranker) at the global level
try:
    reranker = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    print(f"CrossEncoder model loaded successfully on {device}")
except Exception as e:
    st.error(f"Failed to load CrossEncoder model: {str(e)}")


st.set_page_config(
    page_title="DeepGraph RAG-Pro", layout="wide"
)  # ✅ Streamlit configuration

# Custom CSS
st.markdown(
    """
    <style>
        .stApp { background-color: #f4f4f9; }
        h1 { color: #00FF99; text-align: center; }
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 10px 0; }
        .stChatMessage.user { background-color: #e8f0fe; }
        .stChatMessage.assistant { background-color: #d1e7dd; }
        .stButton>button { background-color: #00AAFF; color: white; }
    </style>
""",
    unsafe_allow_html=True,
)


# Manage Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_pipeline" not in st.session_state:
    st.session_state.retrieval_pipeline = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


with st.sidebar:  # 📁 Sidebar
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and not st.session_state.documents_loaded:
        with st.spinner("Processing documents..."):
            process_documents(
                uploaded_files, reranker, EMBEDDINGS_MODEL, OLLAMA_BASE_URL
            )
            st.success("Documents processed!")

    st.markdown("---")
    st.header("⚙️ RAG Settings")

    st.session_state.rag_enabled = st.checkbox("Enable RAG", value=True)
    st.session_state.enable_hyde = st.checkbox("Enable HyDE", value=True)
    st.session_state.enable_reranking = st.checkbox(
        "Enable Neural Reranking", value=True
    )
    st.session_state.enable_graph_rag = st.checkbox("Enable GraphRAG", value=True)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.20, 0.05)
    st.session_state.max_contexts = st.slider("Max Contexts", 1, 10, 5)

    # Smart context adjustment info
    st.info(
        "💡 **Smart Context Adjustment**: The system automatically increases contexts for list/count queries"
    )

    # Chunking best practices info
    with st.expander("📚 Text Chunking Best Practices"):
        st.markdown(
            """
        ### 🎯 **Chunking Strategy Guide**
        
        **Balanced (Default)**: 600 chars, good for most queries
        - ✅ General Q&A, explanations, mixed content
        
        **Detailed**: 400 chars, precise retrieval
        - ✅ Fact extraction, specific details, technical specs
        
        **Contextual**: 800 chars, preserves context
        - ✅ Summaries, complex reasoning, narrative content
        
        **Count Optimized**: 300 chars, item-level precision
        - ✅ Lists, counts, catalogs, structured data
        
        ### 📏 **Size Guidelines**
        - **Small chunks**: Better precision, may lose context
        - **Large chunks**: Better context, may be unfocused
        - **Overlap**: 25% recommended for context continuity
        """
        )

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # 🚀 Footer (Bottom Right in Sidebar) For some Credits :)
    st.sidebar.markdown(
        """
        <div style="position: absolute; top: 20px; right: 10px; font-size: 12px; color: gray;">
            <b>Developed by:</b> Henry &copy; All Rights Reserved 2025
        </div>
    """,
        unsafe_allow_html=True,
    )

# 💬 Chat Interface
st.title("🤖 DeepGraph RAG-Pro")
st.caption(
    "Advanced RAG System with GraphRAG, Hybrid Retrieval, Neural Re-Ranking and Chat History"
)

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    chat_history = "\n".join(
        [msg["content"] for msg in st.session_state.messages[-5:]]
    )  # Last 5 messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # 🚀 Enhanced Context Building
        context = ""
        intent = "explanation"  # default intent

        if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
            try:
                # Detect query intent for better processing
                intent = query_processor.detect_intent(prompt)
                st.write(f"🎯 **Query Intent Detected**: {intent.upper()}")

                # Retrieve documents using enhanced pipeline
                docs = retrieve_documents(prompt, OLLAMA_API_URL, EMBEDDINGS_MODEL, chat_history)

                # Prepare enhanced context using post-processor
                context = result_processor.prepare_context_for_llm(docs, prompt, intent)

                st.write(
                    f"📄 **Context Prepared**: {len(docs)} documents, {len(context)} characters"
                )

            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")

        # 🚀 Enhanced Prompt Generation based on Intent
        system_prompt = result_processor.generate_enhanced_prompt(
            prompt, context, chat_history, intent
        )

        # Stream response
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL,
                "prompt": system_prompt,
                "stream": True,
                "options": {
                    "temperature": st.session_state.temperature,  # Use dynamic user-selected value
                    "num_ctx": 4096,
                },
            },
            stream=True,
        )
        try:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode())
                    token = data.get("response", "")
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

                    # Stop if we detect the end token
                    if data.get("done", False):
                        break

            response_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            st.error(f"Generation error: {str(e)}")
            st.session_state.messages.append(
                {"role": "assistant", "content": "Sorry, I encountered an error."}
            )
