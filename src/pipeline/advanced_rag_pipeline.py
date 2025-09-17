"""
Pipeline RAG avanzata che integra tutte le tecniche:
- Document Processing (Docling)
- Semantic Chunking
- HyPE (Hypothetical Prompt Embeddings)
- Fusion Retrieval
- Gemini Generation
"""
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from config.config import RAGConfig, get_default_config
from src.document_processing.docling_processor import DoclingProcessor, ProcessedDocument
from src.chunking.semantic_chunker import AdvancedSemanticChunker, SemanticChunk
from src.query_handling.hype_processor import HyPEProcessor, EnrichedChunk
from src.retrieval.fusion_retriever import FusionRetriever, RetrievalResult
from src.generation.gemini_generator import GeminiGenerator, GenerationResult


@dataclass
class RAGResponse:
    """Risposta completa del sistema RAG"""
    query: str
    answer: str
    sources: List[str]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]
    retrieval_results: List[RetrievalResult]
    generation_result: GenerationResult


@dataclass
class PipelineState:
    """Stato della pipeline per caching e debugging"""
    processed_documents: List[ProcessedDocument]
    semantic_chunks: List[SemanticChunk]
    enriched_chunks: List[EnrichedChunk]
    indices_built: bool
    last_updated: datetime


class AdvancedRAGPipeline:
    """
    Pipeline RAG avanzata che combina tutte le tecniche implementate
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Inizializza i componenti
        self._initialize_components()
        
        # Stato della pipeline
        self.state: Optional[PipelineState] = None
        self.cache_file = Path(self.config.cache_dir) / "pipeline_state.pkl"
        
        self.logger.info("Advanced RAG Pipeline inizializzata")
    
    def _initialize_components(self):
        """Inizializza tutti i componenti della pipeline"""
        try:
            # Document Processor
            self.doc_processor = DoclingProcessor(self.config.document_processing)
            
            # Semantic Chunker
            self.chunker = AdvancedSemanticChunker(self.config.chunking)
            
            # HyPE Processor
            self.hype_processor = HyPEProcessor(self.config.hype)
            
            # Fusion Retriever
            self.retriever = FusionRetriever(self.config.fusion_retrieval)
            
            # Gemini Generator
            if not self.config.google_api_key:
                raise ValueError("Google API key richiesta per Gemini")
            
            self.generator = GeminiGenerator(
                self.config.generation, 
                self.config.google_api_key
            )
            
            self.logger.info("Tutti i componenti inizializzati con successo")
            
        except Exception as e:
            self.logger.error(f"Errore nell'inizializzazione dei componenti: {e}")
            raise
    
    def process_documents(self, document_paths: List[str], 
                         force_reprocess: bool = False) -> List[ProcessedDocument]:
        """
        Processa i documenti attraverso tutta la pipeline di preparazione
        
        Args:
            document_paths: Lista di percorsi ai documenti da processare
            force_reprocess: Se forzare il riprocessamento anche se esiste cache
            
        Returns:
            List[ProcessedDocument]: Documenti processati
        """
        self.logger.info(f"Processando {len(document_paths)} documenti")
        
        # Controlla se esiste cache valida
        if not force_reprocess and self._load_cache():
            self.logger.info("Cache caricata, saltando il processing")
            return self.state.processed_documents
        
        start_time = datetime.now()
        
        # Step 1: Document Processing
        self.logger.info("Step 1: Document Processing con Docling")
        processed_docs = []
        
        for doc_path in document_paths:
            try:
                if os.path.isfile(doc_path):
                    doc = self.doc_processor.process_document(doc_path)
                    processed_docs.append(doc)
                elif os.path.isdir(doc_path):
                    docs = self.doc_processor.process_directory(doc_path)
                    processed_docs.extend(docs)
                else:
                    self.logger.warning(f"Percorso non valido: {doc_path}")
            except Exception as e:
                self.logger.error(f"Errore nel processare {doc_path}: {e}")
        
        if not processed_docs:
            raise ValueError("Nessun documento processato con successo")
        
        # Step 2: Semantic Chunking
        self.logger.info("Step 2: Semantic Chunking")
        all_chunks = []
        
        for doc in processed_docs:
            try:
                chunks = self.chunker.chunk_text(
                    doc.content, 
                    metadata={
                        "source_file": doc.source_path,
                        "document_title": doc.metadata.get("title", "Unknown"),
                        **doc.metadata
                    }
                )
                all_chunks.extend(chunks)
            except Exception as e:
                self.logger.error(f"Errore nel chunking di {doc.source_path}: {e}")
        
        # Step 3: HyPE Processing
        self.logger.info("Step 3: HyPE Processing (Hypothetical Prompt Embeddings)")
        enriched_chunks = self.hype_processor.process_chunks(all_chunks, use_parallel=True)
        
        # Step 4: Build Retrieval Indices
        self.logger.info("Step 4: Costruzione indici per Fusion Retrieval")
        self.retriever.build_indices(enriched_chunks)
        
        # Salva lo stato
        self.state = PipelineState(
            processed_documents=processed_docs,
            semantic_chunks=all_chunks,
            enriched_chunks=enriched_chunks,
            indices_built=True,
            last_updated=datetime.now()
        )
        
        # Salva cache
        self._save_cache()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Processing completato in {processing_time:.2f} secondi")
        
        # Log statistiche
        self._log_processing_statistics()
        
        return processed_docs
    
    def query(self, query: str, top_k: Optional[int] = None, 
              include_metadata: bool = True) -> RAGResponse:
        """
        Esegue una query completa attraverso la pipeline RAG
        
        Args:
            query: Domanda dell'utente
            top_k: Numero di risultati da recuperare
            include_metadata: Se includere metadati dettagliati
            
        Returns:
            RAGResponse: Risposta completa del sistema
        """
        if not self.state or not self.state.indices_built:
            raise ValueError("Pipeline non inizializzata. Chiamare process_documents() prima.")
        
        start_time = datetime.now()
        self.logger.info(f"Processando query: '{query[:100]}...'")
        
        try:
            # Step 1: Retrieval
            retrieval_results = self.retriever.retrieve(query, top_k)
            
            if not retrieval_results:
                self.logger.warning("Nessun risultato trovato per la query")
                return self._create_empty_response(query, start_time)
            
            # Step 2: Generation
            generation_result = self.generator.generate_answer(query, retrieval_results)
            
            # Calcola tempo totale
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepara metadati
            metadata = {
                "processing_time_seconds": processing_time,
                "num_chunks_retrieved": len(retrieval_results),
                "num_documents_in_index": len(self.state.processed_documents),
                "pipeline_version": "1.0",
                "timestamp": datetime.now().isoformat()
            }
            
            if include_metadata:
                metadata.update({
                    "retrieval_stats": self.retriever.get_retrieval_statistics(retrieval_results),
                    "chunking_stats": self.chunker.get_chunk_statistics(self.state.semantic_chunks),
                    "hype_stats": self.hype_processor.get_processing_statistics(self.state.enriched_chunks)
                })
            
            # Crea risposta
            response = RAGResponse(
                query=query,
                answer=generation_result.answer,
                sources=generation_result.sources_used,
                confidence=generation_result.confidence,
                processing_time=processing_time,
                metadata=metadata,
                retrieval_results=retrieval_results,
                generation_result=generation_result
            )
            
            self.logger.info(f"Query processata in {processing_time:.2f} secondi")
            return response
            
        except Exception as e:
            self.logger.error(f"Errore nel processare la query: {e}")
            return self._create_error_response(query, str(e), start_time)
    
    def batch_query(self, queries: List[str], **kwargs) -> List[RAGResponse]:
        """Processa una lista di query in batch"""
        self.logger.info(f"Processando {len(queries)} query in batch")
        
        responses = []
        for i, query in enumerate(queries, 1):
            self.logger.debug(f"Processando query {i}/{len(queries)}")
            try:
                response = self.query(query, **kwargs)
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Errore nella query {i}: {e}")
                error_response = self._create_error_response(query, str(e), datetime.now())
                responses.append(error_response)
        
        return responses
    
    def get_document_summary(self, summary_type: str = "comprehensive") -> GenerationResult:
        """Genera un riassunto di tutti i documenti processati"""
        if not self.state or not self.state.enriched_chunks:
            raise ValueError("Nessun documento processato")
        
        # Seleziona i chunk più rappresentativi
        top_chunks = []
        for enriched_chunk in self.state.enriched_chunks[:10]:  # Top 10 chunks
            # Crea un RetrievalResult fittizio per il summary
            result = RetrievalResult(
                chunk_id=enriched_chunk.original_chunk.chunk_id,
                content=enriched_chunk.original_chunk.content,
                score=1.0,
                vector_score=1.0,
                bm25_score=1.0,
                metadata=enriched_chunk.original_chunk.metadata,
                rank=len(top_chunks) + 1
            )
            top_chunks.append(result)
        
        return self.generator.generate_summary(top_chunks, summary_type)
    
    def _create_empty_response(self, query: str, start_time: datetime) -> RAGResponse:
        """Crea una risposta vuota quando non ci sono risultati"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RAGResponse(
            query=query,
            answer="Mi dispiace, non ho trovato informazioni rilevanti per rispondere alla tua domanda.",
            sources=[],
            confidence=0.0,
            processing_time=processing_time,
            metadata={"result_type": "empty"},
            retrieval_results=[],
            generation_result=GenerationResult(
                answer="Nessun risultato trovato",
                sources_used=[],
                confidence=0.0,
                metadata={},
                generation_stats={}
            )
        )
    
    def _create_error_response(self, query: str, error: str, start_time: datetime) -> RAGResponse:
        """Crea una risposta di errore"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RAGResponse(
            query=query,
            answer=f"Si è verificato un errore nel processare la tua domanda: {error}",
            sources=[],
            confidence=0.0,
            processing_time=processing_time,
            metadata={"result_type": "error", "error": error},
            retrieval_results=[],
            generation_result=GenerationResult(
                answer=f"Errore: {error}",
                sources_used=[],
                confidence=0.0,
                metadata={},
                generation_stats={}
            )
        )
    
    def _load_cache(self) -> bool:
        """Carica lo stato dalla cache"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.state = pickle.load(f)
                
                # Ricostruisci gli indici
                if self.state.enriched_chunks:
                    self.retriever.build_indices(self.state.enriched_chunks)
                    self.state.indices_built = True
                
                self.logger.info("Cache caricata con successo")
                return True
        except Exception as e:
            self.logger.warning(f"Errore nel caricare la cache: {e}")
        
        return False
    
    def _save_cache(self):
        """Salva lo stato nella cache"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.state, f)
            self.logger.info("Cache salvata con successo")
        except Exception as e:
            self.logger.warning(f"Errore nel salvare la cache: {e}")
    
    def _log_processing_statistics(self):
        """Log delle statistiche di processing"""
        if not self.state:
            return
        
        stats = {
            "documenti_processati": len(self.state.processed_documents),
            "chunks_semantici": len(self.state.semantic_chunks),
            "chunks_arricchiti": len(self.state.enriched_chunks),
            "domande_ipotetiche_totali": sum(
                len(chunk.hypothetical_questions) 
                for chunk in self.state.enriched_chunks
            )
        }
        
        self.logger.info("=== STATISTICHE PROCESSING ===")
        for key, value in stats.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("==============================")
    
    def clear_cache(self):
        """Pulisce la cache"""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
            self.state = None
            self.logger.info("Cache pulita")
        except Exception as e:
            self.logger.error(f"Errore nel pulire la cache: {e}")


def create_rag_pipeline(config: Optional[RAGConfig] = None) -> AdvancedRAGPipeline:
    """Factory function per creare una pipeline RAG"""
    return AdvancedRAGPipeline(config)
