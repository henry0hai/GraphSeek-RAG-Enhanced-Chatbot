# 🚀 **GraphSeek RAG Studio - Advanced Retrieval-Augmented Generation Platform**
**(Free, Private, Local PC Installation ~ Almost No Internet Required)**

> **Features:** BM25, FAISS, Neural Reranking, HyDE, GraphRAG, Chat Memory, Adaptive Chunking, Re-Ranking

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
3. **Smart Context**: Automatically selects the best context based on query intent.
4. **Preprocessing:** Cleans and prepares, detects Intent and Context based on user query.
5. **Hybrid Retrieval:** Combines BM25 and FAISS to fetch the most relevant text chunks.
6. **GraphRAG Processing:** Builds a knowledge graph for deeper context and relationships.
7. **Neural Reranking:** Uses a cross-encoder to reorder retrieved chunks by relevance.
8. **Query Expansion (HyDE):** Generates hypothetical answers to expand your query for better recall.
9. **Chat Memory Integration:** Maintains context by referencing previous user messages.
10. **Using AI Model to Generate:** Produces the final answer based on top-ranked chunks.

---

## **🛠️ Installation & Setup**

### **1 Python/venv Installation**

> **Using Python Version >= 3.11 is recommended.**

```bash
git clone git@github.com:henry0hai/GraphSeek-RAG-Enhanced-Chatbot.git
cd GraphSeek-RAG-Enhanced-Chatbot

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

or I recommend using Conda:


```bash
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
exec zsh
```

Create a New Conda Environment (with compatible Python)

```bash
conda create -n graphseek-faiss python=3.11
conda activate graphseek-faiss
conda install -c pytorch faiss-gpu
pip install --upgrade pip
pip install -r requirements.txt
```

For MacOS users, you can also use Homebrew to install Python and set up a virtual environment:

```bash
pip install faiss-cpu
pip install mlx mlx-lm mlx-metal
```

### **2 Download & Set Up Ollama**

- Download Ollama: [https://ollama.com/](https://ollama.com/)
- Pull required models:
  ```bash
  ollama pull deepseek-r1:7b
  ollama pull gpt-oss:20b # Optional, for better computer hardware
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


> Note: If you want to run Ollama in a separate container, ensure the `ollama` service is running before starting the chatbot service. You can use the `docker-compose up` command to start both services together.
> Note: I don't have time to test the Docker setup, so please report any issues you encounter. For me, the Ollama service works fine on the host machine, and the chatbot runs in a separate container.
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

**Enjoy building knowledge graphs, maintaining conversation memory, and harnessing powerful local LLM inference—all from your own machine.**

---

> **Note:** Inherit from on the original [DeepSeek-RAG-Chatbot](https://github.com/SaiAkhil066/DeepSeek-RAG-Chatbot.git)

---

## ** Future Enhancements**

- **Contextual Definitions:** Provide definitions for context keywords, if not found in the document, then process like a normal query, without using related context from the document.
- **GraphRAG Enhancements:** Improve graph chunking and retrieval logic.
- **Dynamic Context Handling:** Enhance automatic context adjustment based on query intent more intelligently.
