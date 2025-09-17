"""
Embeddings Provider centralizzato per tutto il sistema.
Permette di evitare duplicazioni e cambi rapidi del backend embeddings.
"""
from __future__ import annotations
import os
import threading
from dataclasses import dataclass
from typing import List, Optional

from langchain_openai.embeddings import OpenAIEmbeddings


@dataclass
class EmbeddingsConfig:
    """Configurazione per il provider di embeddings"""
    provider: str = "openai" 
    model: str = "text-embedding-3-small"
    api_key_env: str = "OPENAI_API_KEY"

class EmbeddingsProvider:
    """
    Provider centralizzato di embeddings con lazy init e thread-safety.
    Espone:
    - get_langchain_embeddings(): istanza compatibile LangChain
    - embed_texts(texts)
    - embed_query(text)
    """
    _instance_lock = threading.Lock()
    _singleton: Optional[EmbeddingsProvider] = None

    def __init__(self, config: EmbeddingsConfig):
        self.config = config
        self._lc_embeddings: Optional[OpenAIEmbeddings] = None

    @classmethod
    def get(cls, config: EmbeddingsConfig) -> EmbeddingsProvider:
        """Singleton opzionale per garantire riuso dello stesso provider"""
        with cls._instance_lock:
            if cls._singleton is None:
                cls._singleton = EmbeddingsProvider(config)
            return cls._singleton

    def _ensure_initialized(self):
        if self._lc_embeddings is not None:
            return
        if self.config.provider == "openai":
            api_key = os.getenv(self.config.api_key_env)
            if not api_key:
                raise ValueError(
                    f"API key non trovata in env var {self.config.api_key_env} per embeddings OpenAI"
                )
            self._lc_embeddings = OpenAIEmbeddings(model=self.config.model)
        else:
            raise NotImplementedError(f"Provider embeddings non supportato: {self.config.provider}")

    def get_langchain_embeddings(self):
        """Ritorna un'istanza compatibile con LangChain (SemanticChunker, FAISS, ecc.)"""
        self._ensure_initialized()
        return self._lc_embeddings

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self._ensure_initialized()
        return self._lc_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        self._ensure_initialized()
        return self._lc_embeddings.embed_query(text)
