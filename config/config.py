"""
Configurazione per il sistema RAG avanzato
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DocumentProcessingConfig:
    """Configurazione per il processing dei documenti"""
    input_dir: str = "data"
    output_dir: str = "processed"
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".pdf", ".docx", ".txt", ".md"]


@dataclass
class ChunkingConfig:
    """Configurazione per il semantic chunking"""
    breakpoint_threshold_type: str = "percentile"  # percentile, standard_deviation, interquartile
    breakpoint_threshold_amount: int = 85
    embedding_model: str = "embed-multilingual-light-v3.0"
    min_chunk_size: int = 100
    max_chunk_size: int = 2000


@dataclass
class HyPEConfig:
    """Configurazione per HyPE (Hypothetical Prompt Embeddings)"""
    num_hypothetical_questions: int = 3
    language_model: str = "gemini-1.5-flash"
    embedding_model: str = "embed-multilingual-light-v3.0"
    temperature: float = 0.7
    max_tokens: int = 150


@dataclass
class FusionRetrievalConfig:
    """Configurazione per Fusion Retrieval"""
    vector_weight: float = 0.7  # alpha parameter
    bm25_weight: float = 0.3
    top_k: int = 5
    embedding_model: str = "text-embedding-3-small"


@dataclass
class GenerationConfig:
    """Configurazione per la generazione con Gemini"""
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.3
    max_tokens: int = 1000
    top_p: float = 0.9
    top_k: int = 40


@dataclass
class EmbeddingsSystemConfig:
    """Configurazione per il provider di embeddings centralizzato"""
    provider: str = "huggingface"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    api_key_env: str = "HF_TOKEN"


@dataclass
class RAGConfig:
    """Configurazione principale del sistema RAG"""
    document_processing: DocumentProcessingConfig
    chunking: ChunkingConfig
    hype: HyPEConfig
    fusion_retrieval: FusionRetrievalConfig
    generation: GenerationConfig
    embeddings: EmbeddingsSystemConfig = None
    
    # API Keys
    google_api_key: Optional[str] = None
    
    # Paths
    vector_store_path: str = "vector_stores"
    cache_dir: str = "cache"
    
    def __post_init__(self):
        # Carica le API keys dalle variabili d'ambiente
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Inizializza embeddings config se non fornita
        if self.embeddings is None:
            self.embeddings = EmbeddingsSystemConfig()

        # Crea le directory se non esistono
        os.makedirs(self.vector_store_path, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.document_processing.output_dir, exist_ok=True)


def get_default_config() -> RAGConfig:
    """Restituisce la configurazione di default"""
    return RAGConfig(
        document_processing=DocumentProcessingConfig(),
        chunking=ChunkingConfig(),
        hype=HyPEConfig(),
        fusion_retrieval=FusionRetrievalConfig(),
        generation=GenerationConfig(),
        embeddings=EmbeddingsSystemConfig()
    )
