# rag_pipeline.py
from typing import List
from retriever import RetrieverManager, RetrieverType
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from sentence_transformers import SentenceTransformer
from .llm.llm import BaseLLM, DeepSeekLLM, QwenLLM
import logging
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Orchestrates the end-to-end RAG process"""
    def __init__(self, retriever_manager: RetrieverManager, llm: BaseLLM):
        self.retriever_manager = retriever_manager
        self.llm = llm

    def query(self, query: str, k: int = 1, **generation_kwargs) -> str:
        """Execute full RAG pipeline with configurable parameters"""
        # Retrieve context
        indices = self.retriever_manager.retrieve_top_k(query, k)
        context = self._format_context(indices)
        
        # Build prompt and generate response
        prompt = self._build_prompt(query, context)
        return self.llm.generate(prompt, **generation_kwargs)

    def _format_context(self, indices: List[int]) -> str:
        """Format retrieved documents into context string"""
        return " ".join(
            f"{i+1}. {self.retriever_manager.retriever.data[idx]}\n"
            for i, idx in enumerate(indices)
        )

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        """Construct RAG prompt template"""
        return f"Context: {context}\n\nQuestion: {query}\nAnswer:"

class DocumentManager:
    """Manages document storage and retrieval"""
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.retriever_manager = None

    def initialize_retriever(
        self, 
        retriever_type: RetrieverType,
        embedding_model: SentenceTransformer = None,
        device: str = "cpu"
    ):
        """Initialize and configure the retriever"""
        self.retriever_manager = RetrieverManager.create(
            retriever_type=retriever_type,
            embedding_model=embedding_model,
            device=device
        )
        self.retriever_manager.load_data(self.documents)

# Example usage
if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DOCUMENTS = [
        "The Eiffel Tower was built in 1889.",
        "Paris is the capital of France.",
        "Deep learning is a subset of machine learning.",
        "The Louvre Museum is located in Paris."
    ]

    # Initialize components
    document_manager = DocumentManager(DOCUMENTS)
    document_manager.initialize_retriever(
        RetrieverType.FAISS,
        embedding_model=SentenceTransformer(EMBEDDER_NAME)
    )

    llm = DeepSeekLLM(MODEL_NAME)
    llm.load_model()

    rag_pipeline = RAGPipeline(document_manager.retriever_manager, llm)

    # Execute query
    response = rag_pipeline.query(
        "When was the Eiffel Tower built?",
        k=2,
        max_new_tokens=16384,
        temperature=0.7,
        do_sample=True
    )
    logger.info(response)