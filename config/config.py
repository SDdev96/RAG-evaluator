"""
Configurazione per il sistema RAG avanzato
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


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
    chunking_method: str = "semantic"
    breakpoint_threshold_type: str = "percentile"  # percentile, standard_deviation, interquartile
    breakpoint_threshold_amount: int = 80
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    min_chunk_size: int = 200
    max_chunk_size: int = 2500


@dataclass
class QueryTransformationsConfig:
    """Configurazione per Query Transformations
    
    Args:
        enable_decompose: abilita la decomposizione della query
        enable_rewrite: abilita la riscrittura della query
        enable_expand: abilita l'espansione (step-back) della query
        max_transformations: numero massimo di trasformazioni
        language: lingua
    """
    enable_decompose: bool = False
    enable_rewrite: bool = True
    enable_expand: bool = False
    max_transformations: int = 3
    language: str = "it"


@dataclass
class FusionRetrievalConfig:
    """Configurazione per Fusion Retrieval
    
    Args:
        vector_weight: peso del vettore
        bm25_weight: peso del BM25
        top_k: numero di chunk
        embedding_model: modello di embedding
    """
    vector_weight: float = 0.7  # alpha parameter
    bm25_weight: float = 0.3
    top_k: int = 10
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class GenerationConfig:
    """Configurazione per la generazione con Gemini
    
    Args:
        model_name: nome del modello
        temperature: temperatura del modello
        max_tokens: numero massimo di token
        top_p: top p
        top_k: top k
    """
    model_name: str = "gemini-2.0-flash-lite"
    temperature: float = 0.3
    max_tokens: int = 1000
    top_p: float = 0.9
    top_k: int = 40


@dataclass
class LangfuseConfig:
    """Configuration class for Langfuse settings.
    
    Args:
        public_key: chiave pubblica
        secret_key: chiave segreta
        host: URL del server Langfuse 
        debug: 
    """
    public_key: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
    release: Optional[str] = os.getenv("LANGFUSE_RELEASE")
    host: str = "https://cloud.langfuse.com"
    debug: bool = False
    
    def is_valid(self) -> bool:
        """Check if the configuration has required keys.
        
        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        return bool(self.public_key and self.secret_key)


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
    query_transformations: QueryTransformationsConfig
    fusion_retrieval: FusionRetrievalConfig
    generation: GenerationConfig
    langfuse: LangfuseConfig = None
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

        # Inizializza langfuse config se non fornita
        if self.langfuse is None:
            self.langfuse = LangfuseConfig()

        # Crea le directory se non esistono
        os.makedirs(self.vector_store_path, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.document_processing.output_dir, exist_ok=True)


def get_default_config() -> RAGConfig:
    """Restituisce la configurazione di default"""
    return RAGConfig(
        document_processing=DocumentProcessingConfig(),
        chunking=ChunkingConfig(),
        query_transformations=QueryTransformationsConfig(),
        fusion_retrieval=FusionRetrievalConfig(),
        generation=GenerationConfig(),
        langfuse=LangfuseConfig(),
        embeddings=EmbeddingsSystemConfig()
    )
