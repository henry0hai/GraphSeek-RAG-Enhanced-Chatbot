# 🚀 **GraphSeek RAG Studio - Advanced Retrieval-Augmented Generation Platform**
**(100% Free, Private, Local PC Installation - No Internet Required)**

> **Features:** DeepSeek-7B, BM25, FAISS, Neural Reranking, HyDE, GraphRAG, Chat Memory, Adaptive Chunking

---

## **🔹 Key Features**

- **GraphRAG Integration:** Builds a knowledge graph from your documents for contextual and relational understanding.
- **Neural Reranking:** Uses a cross-encoder for smarter, more relevant chunk ranking.
- **HyDE Query Expansion:** Boosts recall by generating hypothetical answers to expand your queries.
- **Adaptive Chunking:** Select optimal chunking strategies (balanced, detailed, contextual, count-optimized) for your use case.
- **Chat Memory:** Maintains context by referencing chat history for coherent, context-aware responses.
- **Error Handling:** Robust error management for smooth user experience.

---

## **📂 Project Directory Structure**

```
app.py
docker-compose.yml
Dockerfile
LICENSE
README.md
requirements.txt
temp/
utils/
    build_graph.py
    doc_handler.py
    retriever_pipeline.py
    adaptive_chunking.py
```

---

## **⚡ Working Flow**

1. **Upload Documents:** Add PDFs, DOCX, or TXT files via the sidebar.
2. **Choose Chunking Strategy:** Select from balanced, detailed, contextual, or count-optimized chunking for best retrieval.
3. **Hybrid Retrieval:** Combines BM25 and FAISS to fetch the most relevant text chunks.
4. **GraphRAG Processing:** Builds a knowledge graph for deeper context and relationships.
5. **Neural Reranking:** Uses a cross-encoder to reorder retrieved chunks by relevance.
6. **Query Expansion (HyDE):** Generates hypothetical answers to expand your query for better recall.
7. **Chat Memory Integration:** Maintains context by referencing previous user messages.
8. **DeepSeek-7B Generation:** Produces the final answer based on top-ranked chunks.

---

## **🛠️ Installation & Setup**

### **1 Python/venv Installation**

```bash
git clone <your-repo-url>
cd GraphSeek-RAG-Enhanced-Chatbot

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### **2 Download & Set Up Ollama**

- Download Ollama: [https://ollama.com/](https://ollama.com/)
- Pull required models:
  ```bash
  ollama pull deepseek-r1:7b
  ollama pull nomic-embed-text
  ```

### **3 Run the Chatbot**

```bash
ollama serve
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## **🐳 Docker Installation**

### **A) Ollama on Host**

```bash
docker-compose build
docker-compose up
```

### **B) Ollama in Docker**

See `docker-compose.yml` for a two-container setup (Ollama + Chatbot).

---

## **🧠 How It Works**

- **Upload** documents
- **Select** chunking strategy
- **Retrieve** with BM25 + FAISS
- **GraphRAG** for knowledge graph context
- **Re-rank** with neural cross-encoder
- **Expand** queries with HyDE
- **Generate** answers with DeepSeek-7B

---

## **📚 Chunking Best Practices**

- **Balanced:** 600 chars, general Q&A
- **Detailed:** 400 chars, fact extraction
- **Contextual:** 800 chars, summaries
- **Count Optimized:** 300 chars, lists/counts
- **Overlap:** 25% recommended for context continuity

---

## **🔗 Contributing**

- Fork, submit pull requests, or open issues for new features or bug fixes.

---

## **💡 Feedback**

Share your thoughts on [Reddit](https://www.reddit.com/user/akhilpanja/)!

---

**Enjoy building knowledge graphs, maintaining conversation memory, and harnessing powerful local LLM inference—all from your own machine.**


>> **Note:** Inherit from on the original [DeepSeek-RAG-Chatbot](https://github.com/SaiAkhil066/DeepSeek-RAG-Chatbot.git)
