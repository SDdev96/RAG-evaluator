"""
Semantic Chunking implementato seguendo l'approccio di LangChain
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

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

@dataclass
class AdvancedSemanticChunker:
    """
    Chunker semantico avanzato che utilizza embeddings per determinare
    i punti di divisione naturali del testo
    """

    config: ChunkingConfig
    embeddings_provider: EmbeddingsProvider

    # campi non passati al costruttore
    logger: logging.Logger = field(init=False)
    embeddings: Any = field(init=False)
    text_splitter: Any = field(init=False)

    def __post_init__(self):
        """
        Inizializza i campi derivati dopo la creazione automatica da dataclass
        """
        self.logger = logging.getLogger(__name__)
        
        # Provider centralizzato
        self.embeddings = self.embeddings_provider.get_langchain_embeddings()

        # Inizializza il semantic chunker
        self.text_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=self.config.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.config.breakpoint_threshold_amount
        )

        self.logger.info(f"Semantic Chunker inizializzato con {self.config.breakpoint_threshold_type}")
    
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

            # Adattamento dimensioni: fondi i troppo piccoli, dividi i troppo grandi
            chunks: List[SemanticChunk] = []
            pending_content = ""
            pending_start_index = 0
            running_index = 0

            min_size = self.config.min_chunk_size
            max_size = self.config.max_chunk_size

            # Totale caratteri emessi nei chunks finora (per logging)
            total_emitted_chars = 0

            def emit(content: str, start_idx: int, idx: int):
                nonlocal total_emitted_chars
                chunk = SemanticChunk(
                    content=content,
                    metadata={
                        **metadata,
                        "chunk_index": idx,
                        "chunk_size": len(content),
                        "chunking_method": "semantic",
                        "breakpoint_type": self.config.breakpoint_threshold_type,
                    },
                    chunk_id=f"chunk_{idx}",
                    start_index=start_idx,
                    end_index=start_idx + len(content),
                )
                chunks.append(chunk)
                # Aggiorna e logga il totale dei caratteri emessi
                total_emitted_chars += len(content)
                self.logger.info(
                    f"[Chunking] Emesso chunk_index={idx} size={len(content)} total_emitted={total_emitted_chars}"
                )

            next_idx = 0

            for doc in documents:
                seg = doc.page_content
                seg_len = len(seg)

                if pending_content == "":
                    pending_start_index = running_index

                candidate = pending_content + seg if pending_content else seg
                cand_len = len(candidate)

                if cand_len < min_size:
                    self.logger.warning("Chunk troppo piccolo ")
                    # Accumula per fondere con il prossimo segmento
                    pending_content = candidate
                elif cand_len <= max_size:
                    self.logger.info("Chunk dimensione ok")
                    # Dimensione ok: emetti direttamente
                    emit(candidate, pending_start_index, next_idx)
                    next_idx += 1
                    pending_content = ""
                else:
                    self.logger.warning("Chunk troppo grande")
                    # Troppo grande: dividi rispettando [min,max] e confini di frase
                    parts = self._split_text_by_size(candidate, min_size, max_size)
                    # Emetti tutte le parti tranne l'ultima
                    for part in parts[:-1]:
                        emit(part, pending_start_index, next_idx)
                        next_idx += 1
                        pending_start_index += len(part)
                    # Gestisci l'ultima parte: se è piccola, tienila in pending
                    last = parts[-1]
                    if len(last) >= min_size:
                        emit(last, pending_start_index, next_idx)
                        next_idx += 1
                        pending_content = ""
                        pending_start_index += len(last)
                    else:
                        pending_content = last

                running_index += seg_len

            # Se rimane un resto piccolo, fondilo con l'ultimo chunk
            if pending_content:
                if chunks:
                    last_chunk = chunks[-1]
                    last_chunk.content = last_chunk.content + pending_content
                    last_chunk.end_index = last_chunk.start_index + len(last_chunk.content)
                    last_chunk.metadata["chunk_size"] = len(last_chunk.content)
                    # Aggiorna il totale e logga l'operazione di fusione
                    try:
                        # total_emitted_chars esiste nello scope esterno a emit
                        total_emitted_chars += len(pending_content)
                        self.logger.info(
                            f"[Chunking] Fuso resto con ultimo chunk - incremento={len(pending_content)} size_ultimo={len(last_chunk.content)} total_emitted={total_emitted_chars}"
                        )
                    except Exception:
                        # In caso di differenze di scope, evita di interrompere il flusso
                        self.logger.debug("[Chunking] Impossibile aggiornare il totale durante la fusione finale")
                else:
                    emit(pending_content, pending_start_index, next_idx)

            self.logger.info(
                f"Creati {len(chunks)} chunks semantici (adattati tra {min_size} e {max_size} caratteri)"
            )
            return chunks

        except Exception as e:
            self.logger.error(f"Errore nel chunking semantico: {e}")
            raise
    
    def _split_text_by_size(self, content: str, min_size: int, max_size: int) -> List[str]:
        """Divide content in parti tra min_size e max_size privilegiando confini di frase.
        Se non trova confini adatti, taglia a max_size. L'ultima parte troppo piccola
        viene fusa con la precedente.
        """
        parts: List[str] = []
        pos = 0
        n = len(content)

        def find_split(end_limit: int, start_limit: int) -> int:
            window = content[start_limit:end_limit]
            candidates = [window.rfind(x) for x in [". ", "! ", "? ", ".\n", "!\n", "?\n", "\n\n"]]
            best = max(candidates)
            if best == -1:
                return -1
            # aggiungi 2 se separatore ha spazio dopo il punto
            return start_limit + best + (2 if window[best:best+2] in [". ", "! ", "? "] else 1)

        while pos < n:
            end = min(pos + max_size, n)
            length = end - pos
            if min_size <= length <= max_size:
                split_at = find_split(end, pos + min_size)
                if split_at != -1:
                    parts.append(content[pos:split_at].strip())
                    pos = split_at
                else:
                    parts.append(content[pos:end].strip())
                    pos = end
                continue

            split_at = find_split(end, pos + min_size)
            if split_at != -1 and split_at - pos >= min_size:
                parts.append(content[pos:split_at].strip())
                pos = split_at
            else:
                hard_end = pos + max_size
                parts.append(content[pos:hard_end].strip())
                pos = hard_end

        if len(parts) >= 2 and len(parts[-1]) < min_size:
            parts[-2] = (parts[-2] + parts[-1]).strip()
            parts.pop()
        return [p for p in parts if p]
    
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