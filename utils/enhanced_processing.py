import streamlit as st
import re
import spacy
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
import numpy as np

# Try to load spacy model for better NER
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    nlp = None

# Load sentence transformer for semantic analysis
try:
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    SEMANTIC_AVAILABLE = True
except:
    SEMANTIC_AVAILABLE = False
    semantic_model = None


class QueryProcessor:
    """Enhanced query processing for context detection and intent classification"""

    def __init__(self):
        self.intent_patterns = {
            "list": [
                "list",
                "enumerate",
                "what are",
                "show all",
                "give me all",
                "find all",
            ],
            "count": ["how many", "count", "number of", "total", "quantity"],
            "summary": [
                "summarize",
                "summary",
                "overview",
                "main points",
                "key points",
            ],
            "explanation": [
                "explain",
                "why",
                "how",
                "what is",
                "describe",
                "definition",
            ],
            "comparison": ["compare", "difference", "versus", "vs", "contrast"],
            "extraction": ["extract", "find", "get", "retrieve", "show me"],
        }

        # Common domain contexts that might be relevant
        self.domain_contexts = [
            "case study",
            "project",
            "research",
            "analysis",
            "report",
            "study",
            "implementation",
            "solution",
            "methodology",
            "framework",
            "approach",
            "strategy",
            "process",
            "system",
            "model",
            "algorithm",
            "technique",
            "best practice",
            "lesson learned",
            "recommendation",
            "finding",
            "result",
            "outcome",
            "conclusion",
            "insight",
            "trend",
            "pattern",
        ]

    def detect_intent(self, query: str) -> str:
        """Detect the intent of the user query"""
        query_lower = query.lower()

        for intent, patterns in self.intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent

        return "explanation"  # default intent

    def extract_context_keywords(self, query: str) -> List[str]:
        """Extract relevant context keywords from the query"""
        contexts = []
        query_lower = query.lower()

        # Check for domain-specific contexts
        for context in self.domain_contexts:
            if context in query_lower:
                contexts.append(context)

        # Extract entities using spaCy if available
        if SPACY_AVAILABLE and nlp:
            doc = nlp(query)
            entities = [
                ent.text.lower()
                for ent in doc.ents
                if ent.label_
                in [
                    "ORG",
                    "PRODUCT",
                    "WORK_OF_ART",
                    "EVENT",
                    "PERSON",
                    "GPE",
                    "MONEY",
                    "CARDINAL",
                ]
            ]
            contexts.extend(entities)

        # Extract key nouns and phrases
        important_words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
        contexts.extend([word.lower() for word in important_words])

        # For count/list queries, also add common item indicators
        intent = self.detect_intent(query)
        if intent in ["list", "count"]:
            # Add broader search terms for comprehensive retrieval
            item_indicators = [
                "item",
                "element",
                "example",
                "instance",
                "type",
                "kind",
                "category",
                "section",
                "part",
                "component",
            ]
            contexts.extend(item_indicators)

        # Remove duplicates and return
        unique_contexts = list(set(contexts)) if contexts else ["general"]

        # For count/list queries, if we only have 'general', add more broad terms
        if intent in ["list", "count"] and unique_contexts == ["general"]:
            unique_contexts = ["general", "all", "every", "each", "total"]

        return unique_contexts

    def expand_query_with_context(self, query: str, contexts: List[str]) -> str:
        """Expand the query with detected contexts for better retrieval"""
        if not contexts or contexts == ["general"]:
            return query

        context_str = " ".join(contexts)
        return f"{query} {context_str}"


class AdvancedRetriever:
    """Enhanced retrieval with better context search and result combination"""

    def __init__(self):
        self.query_processor = QueryProcessor()

    def search_chunks_by_context(
        self, chunks: List[Any], contexts: List[str], query: str, top_k: int = 10
    ) -> List[Any]:
        """Search chunks using detected contexts with semantic similarity"""
        if not chunks:
            return []

        relevant_chunks = []
        chunk_scores = []

        for chunk in chunks:
            content_lower = chunk.page_content.lower()
            score = 0

            # Context-based scoring
            for context in contexts:
                if context in content_lower:
                    score += 2  # High weight for exact context match

                # Fuzzy matching for related terms
                context_words = context.split()
                for word in context_words:
                    if word in content_lower:
                        score += 0.5

            # Query term matching
            query_words = query.lower().split()
            for word in query_words:
                if len(word) > 2 and word in content_lower:
                    score += 1

            # Always add chunks, even with score 0 for general context
            relevant_chunks.append(chunk)
            chunk_scores.append(score)

        # Sort by score and return top_k
        if relevant_chunks:
            # Create tuples and sort by score
            scored_chunks = list(zip(chunk_scores, relevant_chunks))
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            sorted_chunks = [chunk for score, chunk in scored_chunks]
            return sorted_chunks[:top_k]

        # Fallback: return all chunks if no context matches
        return chunks[:top_k]

    def enhance_graphrag_results(
        self, graph_results: List[str], chunks: List[Any], query: str
    ) -> List[Any]:
        """Better GraphRAG chunk retrieval with semantic matching"""
        if not graph_results or not chunks:
            return []

        graph_chunks = []
        chunk_scores = []

        for chunk in chunks:
            content = chunk.page_content
            score = 0

            # Check for graph entity mentions
            for entity in graph_results:
                if entity.lower() in content.lower():
                    score += 3  # High score for direct entity match

                # Check for partial matches
                entity_words = entity.lower().split()
                for word in entity_words:
                    if len(word) > 2 and word in content.lower():
                        score += 1

            # Boost score if chunk is semantically related to query
            if SEMANTIC_AVAILABLE and semantic_model:
                try:
                    query_embedding = semantic_model.encode([query])
                    chunk_embedding = semantic_model.encode(
                        [content[:500]]
                    )  # Limit text length
                    similarity = np.dot(query_embedding[0], chunk_embedding[0]) / (
                        np.linalg.norm(query_embedding[0])
                        * np.linalg.norm(chunk_embedding[0])
                    )
                    score += similarity * 2  # Add semantic similarity score
                except:
                    pass  # Skip if semantic analysis fails

            if score > 0:
                graph_chunks.append(chunk)
                chunk_scores.append(score)

        # Return top chunks sorted by score
        if graph_chunks:
            # Create tuples and sort by score
            scored_chunks = list(zip(chunk_scores, graph_chunks))
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            sorted_chunks = [chunk for score, chunk in scored_chunks]
            return sorted_chunks[:5]  # Return top 5 GraphRAG chunks

        return []

    def improved_reranking(self, query: str, docs: List[Any], reranker) -> List[Any]:
        """Improved re-ranking with NaN handling and fallback scoring"""
        if not docs or not reranker:
            return docs

        try:
            # Prepare pairs for cross-encoder
            pairs = [
                [query, doc.page_content[:512]] for doc in docs
            ]  # Limit text length

            # Get scores from reranker
            scores = reranker.predict(pairs)

            # Handle NaN scores
            valid_scores = []
            valid_docs = []

            for score, doc in zip(scores, docs):
                if not np.isnan(score) and not np.isinf(score):
                    valid_scores.append(score)
                    valid_docs.append(doc)
                else:
                    # Fallback scoring based on query term matching
                    fallback_score = self.calculate_fallback_score(
                        query, doc.page_content
                    )
                    valid_scores.append(fallback_score)
                    valid_docs.append(doc)

            # Sort by scores
            if valid_scores:
                # Create tuples and sort by score
                scored_docs = list(zip(valid_scores, valid_docs))
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                ranked_docs = [doc for score, doc in scored_docs]
                return ranked_docs

        except Exception as e:
            st.warning(f"Re-ranking failed: {str(e)}. Using original order.")

        return docs

    def calculate_fallback_score(self, query: str, content: str) -> float:
        """Calculate fallback score when reranker fails"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        # Jaccard similarity
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))

        return intersection / union if union > 0 else 0.0


class ResultPostProcessor:
    """Post-processing for better result combination and answer preparation"""

    def __init__(self):
        self.query_processor = QueryProcessor()

    def combine_results(
        self, hybrid_docs: List[Any], graph_docs: List[Any], max_contexts: int
    ) -> List[Any]:
        """Intelligently combine hybrid and GraphRAG results"""
        combined_docs = []
        seen_content = set()

        # Prioritize GraphRAG results (they're often more relevant)
        for doc in graph_docs:
            content_hash = hash(
                doc.page_content[:100]
            )  # Use first 100 chars as identifier
            if content_hash not in seen_content:
                combined_docs.append(doc)
                seen_content.add(content_hash)

        # Add hybrid results that aren't duplicates
        for doc in hybrid_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content and len(combined_docs) < max_contexts:
                combined_docs.append(doc)
                seen_content.add(content_hash)

        return combined_docs[:max_contexts]

    def prepare_context_for_llm(self, docs: List[Any], query: str, intent: str) -> str:
        """Prepare optimized context based on query intent"""
        if not docs:
            return ""

        context_parts = []

        # Add intent-specific instructions
        if intent == "list":
            context_parts.append(
                "INSTRUCTION: The user wants a comprehensive list. Extract and enumerate ALL relevant items."
            )
        elif intent == "count":
            context_parts.append(
                "INSTRUCTION: The user wants to count items. List all items and provide the total count."
            )
        elif intent == "summary":
            context_parts.append(
                "INSTRUCTION: The user wants a summary. Provide key points and main themes."
            )

        # Add numbered sources
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Source {i}]: {doc.page_content}")

        return "\n\n".join(context_parts)

    def generate_enhanced_prompt(
        self, query: str, context: str, chat_history: str, intent: str
    ) -> str:
        """Generate enhanced system prompt based on intent and context"""

        base_prompt = f"""You are an expert assistant with access to relevant documents. 

Chat History:
{chat_history}

Query Intent: {intent.upper()}
"""

        if intent == "list":
            instruction = """Your task is to provide a COMPREHENSIVE LIST of all relevant items from the context.
- Extract ALL items that match the user's request
- Number each item clearly
- Provide brief descriptions where available
- Do NOT limit yourself to just a few items - be thorough
- Cite sources using [Source #] notation"""

        elif intent == "count":
            instruction = """Your task is to COUNT and LIST ALL relevant items comprehensively.
- CAREFULLY examine EVERY source provided
- List ALL items you find across ALL sources - do not stop at just a few
- Number each item clearly (1, 2, 3, etc.)
- After listing all items, provide the TOTAL COUNT
- Be exhaustive and thorough - missing items leads to incorrect counts
- If you find similar items in different sources, count them separately
- Cite the source for each item using [Source #] notation
- Double-check your count matches the number of items listed"""

        elif intent == "summary":
            instruction = """Your task is to provide a comprehensive SUMMARY.
- Cover all main points and themes
- Organize information logically
- Include key details and insights
- Cite sources using [Source #] notation"""

        else:
            instruction = """Your task is to provide a detailed, accurate response.
- Use all relevant information from the context
- Provide comprehensive coverage of the topic
- Include supporting details and examples
- Cite sources using [Source #] notation"""

        return f"""{base_prompt}

{instruction}

Context:
{context}

Question: {query}

Answer:"""
