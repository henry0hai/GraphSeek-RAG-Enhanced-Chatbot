"""
Advanced Text Chunking Strategies for RAG Systems

This module provides best practices and adaptive chunking strategies
for different types of documents and use cases.
"""

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from typing import List, Dict, Any
import streamlit as st


class AdaptiveChunker:
    """Advanced chunking with document-type awareness and optimization strategies"""

    def __init__(self):
        # Best practice configurations for different scenarios
        self.chunking_strategies = {
            "balanced": {
                "chunk_size": 600,
                "chunk_overlap": 150,
                "description": "Balanced approach for most documents",
            },
            "detailed": {
                "chunk_size": 400,
                "chunk_overlap": 100,
                "description": "Smaller chunks for detailed analysis and precise retrieval",
            },
            "contextual": {
                "chunk_size": 800,
                "chunk_overlap": 200,
                "description": "Larger chunks preserving more context",
            },
            "count_optimized": {
                "chunk_size": 300,
                "chunk_overlap": 75,
                "description": "Optimized for counting and listing tasks",
            },
        }

    def get_optimal_chunker(
        self, documents: List[Any], strategy: str = "balanced"
    ) -> Any:
        """Get the optimal text splitter based on document characteristics"""

        if strategy not in self.chunking_strategies:
            strategy = "balanced"

        config = self.chunking_strategies[strategy]

        # Analyze document characteristics
        total_length = sum(len(doc.page_content) for doc in documents)
        avg_doc_length = total_length / len(documents) if documents else 0

        st.write(f"📋 **Chunking Strategy**: {strategy.upper()}")
        st.write(
            f"📊 **Document Analysis**: {len(documents)} docs, avg length: {avg_doc_length:.0f} chars"
        )
        st.write(f"⚙️ **Settings**: {config['description']}")

        # Primary splitter: RecursiveCharacterTextSplitter (best for most cases)
        primary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=[
                "\n\n",  # Paragraph breaks (highest priority)
                "\n",  # Line breaks
                ". ",  # Sentence endings
                "! ",  # Exclamations
                "? ",  # Questions
                "; ",  # Semicolons
                ", ",  # Commas
                " ",  # Spaces
                "",  # Character level (last resort)
            ],
            length_function=len,
            is_separator_regex=False,
        )

        return primary_splitter

    def analyze_chunk_quality(self, chunks: List[Any]) -> Dict[str, Any]:
        """Analyze the quality of generated chunks"""
        if not chunks:
            return {}

        chunk_lengths = [len(chunk.page_content) for chunk in chunks]

        analysis = {
            "total_chunks": len(chunks),
            "avg_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_length": min(chunk_lengths),
            "max_length": max(chunk_lengths),
            "std_dev": self._calculate_std_dev(chunk_lengths),
        }

        # Quality indicators
        analysis["quality_score"] = self._calculate_quality_score(analysis)
        analysis["recommendations"] = self._get_recommendations(analysis)

        return analysis

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5

    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate a quality score for chunking (0-100)"""
        score = 100

        # Penalize if chunks are too small or too large
        avg_length = analysis["avg_length"]
        if avg_length < 200:
            score -= 20  # Too small, might lose context
        elif avg_length > 1000:
            score -= 15  # Too large, might be unfocused

        # Penalize high variance (inconsistent chunk sizes)
        if analysis["std_dev"] > avg_length * 0.5:
            score -= 15

        # Penalize if min/max ratio is too extreme
        ratio = (
            analysis["max_length"] / analysis["min_length"]
            if analysis["min_length"] > 0
            else 1
        )
        if ratio > 5:
            score -= 10

        return max(0, score)

    def _get_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Get recommendations for improving chunking"""
        recommendations = []

        if analysis["avg_length"] < 200:
            recommendations.append(
                "Consider increasing chunk_size for better context preservation"
            )

        if analysis["avg_length"] > 1000:
            recommendations.append(
                "Consider decreasing chunk_size for more focused retrieval"
            )

        if analysis["std_dev"] > analysis["avg_length"] * 0.5:
            recommendations.append(
                "High variance in chunk sizes - consider adjusting separators"
            )

        ratio = (
            analysis["max_length"] / analysis["min_length"]
            if analysis["min_length"] > 0
            else 1
        )
        if ratio > 5:
            recommendations.append(
                "Large size variation - consider preprocessing documents"
            )

        if analysis["quality_score"] > 80:
            recommendations.append("✅ Good chunking quality!")

        return recommendations


# Best Practices Documentation
CHUNKING_BEST_PRACTICES = """
## 📚 Text Chunking Best Practices for RAG

### 1. **Chunk Size Guidelines**
- **Small chunks (200-400 chars)**: Best for precise fact retrieval, Q&A
- **Medium chunks (400-800 chars)**: Balanced approach, good for most use cases
- **Large chunks (800-1200 chars)**: Better context preservation, good for summaries

### 2. **Overlap Strategies**
- **Standard overlap**: 20-25% of chunk size
- **High overlap**: 30-40% for critical context preservation
- **Low overlap**: 10-15% for efficiency when storage is a concern

### 3. **Separator Hierarchy**
1. Paragraph breaks (`\\n\\n`) - Preserves semantic structure
2. Line breaks (`\\n`) - Good fallback
3. Sentence endings (`. `, `! `, `? `) - Maintains sentence integrity
4. Punctuation (`;`, `,`) - Clause-level breaks
5. Spaces - Last resort

### 4. **Document-Type Specific Strategies**
- **Academic papers**: Larger chunks (600-800) to preserve arguments
- **Technical docs**: Medium chunks (400-600) for step-by-step procedures
- **Lists/catalogs**: Smaller chunks (200-400) for item-level retrieval
- **Narrative text**: Larger chunks (600-1000) for story continuity

### 5. **Quality Indicators**
- **Consistency**: Similar chunk sizes indicate good splitting
- **Completeness**: No broken sentences or concepts
- **Relevance**: Each chunk should be meaningful standalone
- **Overlap effectiveness**: Related chunks should share important context

### 6. **Performance Considerations**
- **Retrieval speed**: Smaller chunks = faster search
- **Context quality**: Larger chunks = better context
- **Storage efficiency**: Less overlap = smaller vector database
- **Re-ranking effectiveness**: Medium chunks often optimal for cross-encoders
"""
