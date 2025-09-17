"""
Semantic Chunking implementato seguendo l'approccio di LangChain
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document

from config.config import ChunkingConfig
from src.embeddings.provider import EmbeddingsProvider


@dataclass
class SemanticChunk:
    """Rappresenta un chunk semantico
    
    Args:
        content (str): Contenuto del chunk
        metadata (Dict[str, Any]): Metadati associati al chunk
        chunk_id (str): ID univoco del chunk
        start_index (int): Indice di inizio del chunk
        end_index (int): Indice di fine del chunk 
    """
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_index: int
    end_index: int


class AdvancedSemanticChunker:
    """
    Chunker semantico avanzato che utilizza embeddings per determinare
    i punti di divisione naturali del testo
    """
    
    def __init__(self, config: ChunkingConfig, embeddings_provider: EmbeddingsProvider):
        """
        Inizializza il semantic chunker
        
        Args:
            config: Configurazione del chunking
            embeddings_provider: Provider di embeddings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Provider centralizzato
        self.embeddings_provider = embeddings_provider
        self.embeddings = embeddings_provider.get_langchain_embeddings()
        
        # Inizializza il semantic chunker
        self.text_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=config.breakpoint_threshold_type,
            breakpoint_threshold_amount=config.breakpoint_threshold_amount
        )
        
        self.logger.info(f"Semantic Chunker inizializzato con {config.breakpoint_threshold_type}")
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[SemanticChunk]:
        """
        Divide il testo in chunks semantici
        
        Args:
            text: Testo da dividere
            metadata: Metadati opzionali da associare ai chunks
            
        Returns:
            List[SemanticChunk]: Lista di chunks semantici
        """
        if metadata is None:
            metadata = {}
        
        self.logger.info("Inizio chunking semantico del testo")
        
        try:
            # Crea i documenti usando il semantic chunker
            documents = self.text_splitter.create_documents([text])
            
            chunks = []
            current_index = 0
            total_length = 0
            
            for i, doc in enumerate(documents):
                chunk_content = doc.page_content
                
                # Calcola gli indici di inizio e fine
                start_index = current_index
                end_index = start_index + len(chunk_content)
                current_index = end_index
                
                # Verifica dimensioni del chunk
                if len(chunk_content) < self.config.min_chunk_size:
                    total_length += len(chunk_content)
                    self.logger.warning(f"Chunk {i} troppo piccolo ({len(chunk_content)} caratteri). Lunghezza totale: {total_length}")
                    continue
                
                if len(chunk_content) > self.config.max_chunk_size:
                    total_length += len(chunk_content)
                    self.logger.warning(f"Chunk {i} troppo grande ({len(chunk_content)} caratteri). Lunghezza totale: {total_length}")
                    # Suddivide ulteriormente il chunk se necessario
                    sub_chunks = self._split_large_chunk(chunk_content, i, start_index, metadata)
                    chunks.extend(sub_chunks)
                else:
                    total_length += len(chunk_content)
                    self.logger.warning(f"Chunk {i} nella media ({len(chunk_content)} caratteri). Lunghezza totale: {total_length}")
                    # Crea il chunk semantico
                    chunk = SemanticChunk(
                        content=chunk_content,
                        metadata={
                            **metadata,
                            "chunk_index": i,
                            "chunk_size": len(chunk_content),
                            "chunking_method": "semantic",
                            "breakpoint_type": self.config.breakpoint_threshold_type
                        },
                        chunk_id=f"chunk_{i}",
                        start_index=start_index,
                        end_index=end_index
                    )
                    chunks.append(chunk)
            
            self.logger.info(f"Creati {len(chunks)} chunks semantici")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Errore nel chunking semantico: {e}")
            # Fallback al chunking tradizionale
            return self._fallback_chunking(text, metadata)
    
    def _split_large_chunk(self, content: str, base_index: int, base_start_index: int, 
                          metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """Suddivide un chunk troppo grande in chunks più piccoli"""
        chunks = []
        chunk_size = self.config.max_chunk_size
        overlap = min(200, chunk_size // 10)  # 10% di overlap
        
        current_pos = 0
        sub_chunk_index = 0
        
        while current_pos < len(content):
            end_pos = min(current_pos + chunk_size, len(content))
            
            # Cerca un punto di divisione naturale (fine frase)
            if end_pos < len(content):
                # Cerca l'ultimo punto, punto esclamativo o punto interrogativo
                for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                    last_punct = content.rfind(punct, current_pos, end_pos)
                    if last_punct != -1:
                        end_pos = last_punct + len(punct)
                        break
            
            chunk_content = content[current_pos:end_pos].strip()
            
            if chunk_content:
                chunk = SemanticChunk(
                    content=chunk_content,
                    metadata={
                        **metadata,
                        "chunk_index": f"{base_index}_{sub_chunk_index}",
                        "chunk_size": len(chunk_content),
                        "chunking_method": "semantic_split",
                        "parent_chunk": base_index
                    },
                    chunk_id=f"chunk_{base_index}_{sub_chunk_index}",
                    start_index=base_start_index + current_pos,
                    end_index=base_start_index + end_pos
                )
                chunks.append(chunk)
                sub_chunk_index += 1
            
            # Avanza con overlap
            current_pos = end_pos - overlap if end_pos < len(content) else end_pos
        
        return chunks
    
    def _fallback_chunking(self, text: str, metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """Chunking di fallback quando il semantic chunking fallisce"""
        self.logger.warning("Usando chunking di fallback")
        
        chunks = []
        chunk_size = 1000
        overlap = 200
        
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(text):
            end_pos = min(current_pos + chunk_size, len(text))
            
            # Cerca un punto di divisione naturale
            if end_pos < len(text):
                for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                    last_punct = text.rfind(punct, current_pos, end_pos)
                    if last_punct != -1:
                        end_pos = last_punct + len(punct)
                        break
            
            chunk_content = text[current_pos:end_pos].strip()
            
            if chunk_content:
                chunk = SemanticChunk(
                    content=chunk_content,
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk_content),
                        "chunking_method": "fallback"
                    },
                    chunk_id=f"fallback_chunk_{chunk_index}",
                    start_index=current_pos,
                    end_index=end_pos
                )
                chunks.append(chunk)
                chunk_index += 1
            
            current_pos = end_pos - overlap if end_pos < len(text) else end_pos
        
        return chunks
    
    def chunk_documents(self, documents: List[str], 
                       documents_metadata: Optional[List[Dict[str, Any]]] = None) -> List[SemanticChunk]:
        """
        Divide una lista di documenti in chunks semantici
        
        Args:
            documents: Lista di testi da dividere
            documents_metadata: Lista di metadati per ogni documento
            
        Returns:
            List[SemanticChunk]: Lista di tutti i chunks semantici
        """
        if documents_metadata is None:
            documents_metadata = [{}] * len(documents)
        
        all_chunks = []
        
        for i, (doc, metadata) in enumerate(zip(documents, documents_metadata)):
            doc_metadata = {
                **metadata,
                "document_index": i,
                "source_document": f"doc_{i}"
            }
            
            chunks = self.chunk_text(doc, doc_metadata)
            all_chunks.extend(chunks)
        
        self.logger.info(f"Processati {len(documents)} documenti, creati {len(all_chunks)} chunks totali")
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[SemanticChunk]) -> Dict[str, Any]:
        """Calcola statistiche sui chunks creati"""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_characters": sum(chunk_sizes),
            "chunking_methods": list(set(chunk.metadata.get("chunking_method", "unknown") for chunk in chunks))
        }
        
        return stats


def create_semantic_chunker(config: Optional[ChunkingConfig] = None, 
                           embeddings_provider: Optional[EmbeddingsProvider] = None) -> AdvancedSemanticChunker:
    """Factory function per creare un semantic chunker"""
    if config is None:
        config = ChunkingConfig()
    if embeddings_provider is None:
        raise ValueError("EmbeddingsProvider è richiesto per creare AdvancedSemanticChunker")
    return AdvancedSemanticChunker(config, embeddings_provider)