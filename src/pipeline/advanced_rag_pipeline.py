"""
Pipeline RAG avanzata che integra tutte le tecniche:
- Document Processing (Docling)
- Semantic Chunking
- Query Transformations (al posto di HyPE)
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
import numpy as np
import faiss

from config.config import RAGConfig, get_default_config
from src.document_processing.document_processor import DocumentProcessor, ProcessedDocument
from src.chunking.semantic_chunker import AdvancedSemanticChunker, SemanticChunk
from src.query_handling.query_transformations import QueryTransformer
from src.retrieval.fusion_retriever import FusionRetriever, RetrievalResult
from src.generation.gemini_generator import GeminiGenerator, GenerationResult
from src.embeddings.provider import EmbeddingsProvider
from src.telemetry.langfuse_setup import LangfuseManager


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
    summary: str


@dataclass
class PipelineState:
    """Stato della pipeline per caching e debugging"""
    processed_documents: List[ProcessedDocument]
    semantic_chunks: List[SemanticChunk]
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
            # Document Processor (Docling)
            self.doc_processor = DocumentProcessor(self.config.document_processing)
            
            # Semantic Chunker
            self.embeddings_provider = EmbeddingsProvider.get(self.config.embeddings)

            self.chunker = AdvancedSemanticChunker(self.config.chunking, self.embeddings_provider)
            
            # Query Transformer
            self.query_transformer = QueryTransformer(self.config.query_transformations)
            
            # Fusion Retriever
            self.retriever = FusionRetriever(self.config.fusion_retrieval, self.embeddings_provider)
            
            # Gemini Generator
            if not self.config.google_api_key:
                raise ValueError("Google API key richiesta per Gemini")
            
            self.generator = GeminiGenerator(
                self.config.generation, 
                self.config.google_api_key
            )

            # Langfuse initialize
            self.langfuse = LangfuseManager(self.config.langfuse)

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
        
        self.logger.info(f"Processando {len(document_paths)} document{'i' if len(document_paths) > 1 else 'o'}")

        
        # Controlla se esiste cache valida
        # if not force_reprocess and self._load_cache():
        #     self.logger.info("Cache caricata, saltando il processing")
        #     return self.state.processed_documents
        
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
                # 2.a) Prova a caricare chunk già calcolati da chunked_document/
                loaded_chunks = self._load_chunked_artifacts(doc)
                if loaded_chunks is not None and len(loaded_chunks) > 0:
                    self.logger.info(f"Chunk precomputati trovati per {doc.source_path}. Salto il chunking.")
                    print(f"Chunk precomputati trovati per {doc.source_path}. Salto il chunking.")
                    chunks = loaded_chunks
                else:
                    # 2.b) Altrimenti esegui il chunking e salva gli artefatti
                    chunks = self.chunker.chunk_text(
                        doc.content, 
                        metadata={
                            "source_file": doc.source_path,
                            "document_title": doc.metadata.get("title", "Unknown"),
                            **doc.metadata
                        }
                    )
                    # Salva artefatti per-documento (FAISS + metadati/chunk)
                    try:
                        self._save_chunked_artifacts(doc, chunks)
                    except Exception as se:
                        self.logger.warning(f"Impossibile salvare artefatti chunk per {doc.source_path}: {se}")

                all_chunks.extend(chunks)
            except Exception as e:
                self.logger.error(f"Errore nel chunking di {doc.source_path}: {e}")
        
        # Step 3: Costruzione/Caricamento indici per Retrieval
        self.logger.info("Step 3: Preparazione/caricamento indici per Fusion Retrieval")
        index_dir = Path("indexed_chunks")
        
        # 3.a) Prova a caricare indici già esistenti da disco
        loaded_ok = False
        try:
            if self.retriever.load_indices(index_dir):
                loaded_ok = True
                self.logger.info(f"Indici di retrieval caricati da '{index_dir}'.")
                print(f"Indici di retrieval caricati da '{index_dir}'.")
        except Exception as e:
            self.logger.warning(f"Errore nel caricamento degli indici da '{index_dir}': {e}")
            print(f"Errore nel caricamento degli indici da '{index_dir}': {e}")

        # 3.b) Se non presenti, costruiscili e salvali
        if not loaded_ok:
            self.logger.info("Indici non trovati: preparazione in corso...")
            self.retriever.build_indices(all_chunks)
            self.logger.info("Indici costruiti con successo.")
            print("Indici costruiti con successo.")
            # Salva indici costruiti su disco per riuso futuro
            try:
                self.retriever.save_indices(index_dir)
                self.logger.info(f"Indici salvati su '{index_dir}'")
                print(f"Indici salvati su '{index_dir}'")
            except Exception as e:
                self.logger.warning(f"Impossibile salvare gli indici su disco: {e}")
                print(f"Impossibile salvare gli indici su disco: {e}")
        
        # Salva lo stato
        self.state = PipelineState(
            processed_documents=processed_docs,
            semantic_chunks=all_chunks,
            indices_built=True,
            last_updated=datetime.now()
        )
        
        # Ignora il salvataggio della cache (richiesto)
        # self._save_cache()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Processing completato in {processing_time:.2f} secondi")
        
        # Log statistiche
        self._log_processing_statistics()
        
        return processed_docs
    
    def query(self, query: str, top_k: int = get_default_config().fusion_retrieval.top_k, 
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
        self.logger.info(f"Processando query: '{query}'")
        
        try:
            # Step 1: Query Transformations + Retrieval
            variants = self.query_transformer.transform(query)
            queries_for_retrieval = variants if variants else [query]
            retrieval_results = self.retriever.retrieve_multi(queries_for_retrieval, top_k)
            
            if not retrieval_results:
                self.logger.warning("Nessun risultato trovato per la query")
                return self._create_empty_response(query, start_time)
            
            # Step 2: Generation
            generation_result = self.generator.generate_answer(query, retrieval_results)
            
            # Calcola tempo totale
            processing_time = (datetime.now() - start_time).total_seconds()

            # Step 3: Summary
            summary = self.generator.generate_summary_from_llm_response(query, generation_result.answer)
            
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
                    "query_variants": variants,
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
                generation_result=generation_result,
                summary=summary
            )
            
            self.logger.info(f"Query processata in {processing_time:.2f} secondi")
            return response
            
        except Exception as e:
            self.logger.error(f"Errore nel processare la query: {e}")
            return self._create_error_response(query, str(e), start_time)

    def query_with_langfuse_simple_cm(self, 
                          query: str, 
                          top_k: int = None,
                          include_metadata: bool = True,
                          user_id: Optional[str] = "user_id",
                          session_id: Optional[str] = "session_id",
                          tags: Optional[List[str]] = []) -> RAGResponse:
        """
        Esegue una query completa attraverso la pipeline RAG e la traccia attraverso Langfuse
        """
        
        if top_k is None:
            top_k = get_default_config().fusion_retrieval.top_k
        
        if not self.state or not self.state.indices_built:
            raise ValueError("Pipeline non inizializzata. Chiamare process_documents() prima.")
        
        langfuse_client = self.langfuse.get_client()

        if langfuse_client:
            start_time = datetime.now()
            self.logger.info(f"Processando query: '{query}'")

            try:
                # ROOT: rag_pipeline
                with langfuse_client.start_as_current_span(
                    name="rag_pipeline", 
                    input=query,
                ) as rag_pipeline:

                    rag_pipeline.update_trace(
                        user_id=user_id,
                        session_id=session_id,
                        version="test_version",
                        tags= tags or ["rag", "nested"],
                    )

                    variants, llm_datas = self.query_transformer.transform(query)

                    # 1. query_transformation
                    with rag_pipeline.start_as_current_observation(
                        name="query_transformation",
                        as_type="generation",
                        model=self.generator.config.model_name if hasattr(self.generator.config, 'model_name') else "llm_generator",
                        input=llm_datas['prompt_used'],
                        
                    ) as query_transf:

                        # Test scores
                        # query_transf.score(name="custom_score_test", value=0.8, data_type="NUMERIC")
                        # query_transf.score_trace(name="user_feedback_rating", value="positive", data_type="CATEGORICAL")

                        query_transf.update(
                            output=llm_datas['transformations'][1]["query"],
                            usage_details = {"input_tokens": llm_datas["input_tokens"], 
                                             "output_tokens": llm_datas["output_tokens"], 
                                             "total_tokens": llm_datas["input_tokens"] + llm_datas["output_tokens"]},
                            metadata = {"prompt usato": llm_datas['prompt_used'], 
                                        "queries": llm_datas['transformations'],
                                        "variants": variants},
                        )

                        queries_for_retrieval = variants if variants else [query] 

                        # 2. Retrieval
                        with query_transf.start_as_current_observation(
                            name="retrieval",
                            as_type="retriever",
                            input="\n".join(queries_for_retrieval),
                        ) as retrieval_obs:

                            retrieval_results = self.retriever.retrieve_multi(queries_for_retrieval, top_k)
                            context_text = "\n\n".join([f"--- ({result.chunk_id}) ---\n{result.content}" for result in retrieval_results])

                            retrieval_obs.update(
                                output=context_text,
                                metadata = retrieval_results
                            )

                            generation_result = self.generator.generate_answer(queries_for_retrieval, retrieval_results)

                            # 3. Answer Generation
                            with retrieval_obs.start_as_current_observation(
                                name="answer_generation",
                                as_type="generation",
                                model=self.generator.config.model_name if hasattr(self.generator.config, 'model_name') else "llm_generator",
                                input=generation_result.prompt,
                            ) as answer_gen:

                                answer_gen.update(
                                    output=generation_result.answer,
                                    metadata = {"generation_result": generation_result, "context_text": context_text},
                                    usage_details = {"input_tokens": generation_result.metadata[2]["input_tokens"], 
                                                     "output_tokens": generation_result.metadata[2]["output_tokens"], 
                                                     "total_tokens": generation_result.metadata[2]["input_tokens"] + generation_result.metadata[2]["output_tokens"]},
                                )

                                summary_result = self.generator.generate_summary_from_llm_response(queries_for_retrieval, generation_result.answer)

                                # 4. Summary Generation
                                with answer_gen.start_as_current_observation(
                                    name="summary_generation",
                                    as_type="generation",
                                    model=self.generator.config.model_name if hasattr(self.generator.config, 'model_name') else "llm_generator",
                                    input=summary_result.prompt
                                ) as summary_gen:

                                    summary_gen.update(
                                        output=summary_result.answer,
                                        metadata = summary_result,
                                        usage_details = {"input_tokens": summary_result.metadata[2]["input_tokens"], 
                                                         "output_tokens": summary_result.metadata[2]["output_tokens"], 
                                                         "total_tokens": summary_result.metadata[2]["input_tokens"] + summary_result.metadata[2]["output_tokens"]},
                                    )


                                # Fine pipeline → aggiorno rag_pipeline
                                processing_time = (datetime.now() - start_time).total_seconds()

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
                                        "query_variants": variants,
                                    })

                                response = RAGResponse(
                                    query=query,
                                    answer=generation_result.answer,
                                    sources=generation_result.sources_used,
                                    confidence=generation_result.confidence,
                                    processing_time=processing_time,
                                    metadata=metadata,
                                    retrieval_results=retrieval_results,
                                    generation_result=generation_result,
                                    summary=summary_result
                                )

                                rag_pipeline.update(
                                    output={
                                        "answer": response.answer,
                                        "retrieval_results": response.retrieval_results,
                                        "num_sources": response.sources,
                                        "generation_result": response.generation_result,
                                        "summary": response.summary,
                                        "confidence": response.confidence,
                                        "processing_time": response.processing_time,
                                    },
                                    metadata=response.metadata
                                )

                                self.logger.info(f"Query processata in {processing_time:.2f} secondi")
                                return response

            except Exception as e:
                self.logger.error(f"Errore nel processare la query: {e}")
                error_response = self._create_error_response(query, str(e), start_time)
                
                if langfuse_client:
                    with langfuse_client.start_as_current_span(
                        name="rag_query_pipeline_error",
                        input={"user_query": query},
                    ) as error_span:
                        error_span.update_trace(
                            user_id=user_id,
                            session_id=session_id,
                            tags=["error"] + (tags or [])
                        )
                        error_span.update(
                            output={"error": str(e)},
                            metadata={"error_type": type(e).__name__}
                        )
                
                return error_response
        else:
            return self.query(query, top_k, include_metadata)


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
        if not self.state or not self.state.semantic_chunks:
            raise ValueError("Nessun documento processato")
        
        # Seleziona i chunk più rappresentativi
        top_chunks = []
        for chunk in self.state.semantic_chunks[:get_default_config().fusion_retrieval.top_k]:
            result = RetrievalResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=1.0,
                vector_score=1.0,
                bm25_score=1.0,
                metadata=chunk.metadata,
                rank=len(top_chunks) + 1
            )
            top_chunks.append(result)
        
        return self.generator.generate_summary_from_chunks(top_chunks, summary_type)
    
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
                metadata=[
                    {},  # RAG_metadata vuoto
                    {},  # response_metadata vuoto
                    {}   # usage_metadata vuoto
                ],
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
                metadata=[
                    {},  # RAG_metadata vuoto
                    {},  # response_metadata vuoto
                    {}   # usage_metadata vuoto
                ],
                generation_stats={}
            )
        )
    
    def _log_processing_statistics(self):
        """Log delle statistiche di processing"""
        if not self.state:
            return
        
        stats = {
            "documenti_processati": len(self.state.processed_documents),
            "chunks_semantici": len(self.state.semantic_chunks),
        }
        
        self.logger.info("=== STATISTICHE PROCESSING ===")
        for key, value in stats.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("==============================")
    
    def _save_chunked_artifacts(self, doc: ProcessedDocument, chunks: List[SemanticChunk]):
        """Salva prima del retrieval gli artefatti per ciascun documento:
        - Indice vettoriale FAISS dei soli chunk del documento (xx.faiss)
        - File pickle con metadati e contenuti originali dei chunk (xx.pkl)
        Directory di destinazione: project_root/chunked_document/
        """
        if not chunks:
            return

        # Prepara directory e nomi file basati sul nome del file processato (xx.md -> xx)
        base_dir = Path("chunked_document")
        base_dir.mkdir(parents=True, exist_ok=True)

        # Preferisci l'output_path (markdown) se presente, altrimenti usa source_path
        try:
            stem = Path(doc.output_path).stem if getattr(doc, "output_path", None) else Path(doc.source_path).stem
        except Exception:
            stem = Path(doc.source_path).stem

        faiss_path = base_dir / f"{stem}.faiss"
        pkl_path = base_dir / f"{stem}.pkl"

        # 1) Calcola embeddings dei chunk del documento
        texts = [c.content for c in chunks]
        embeddings = self.embeddings_provider.embed_texts(texts)
        if not embeddings:
            raise ValueError("Nessun embedding generato per i chunk")

        # 2) Costruisci indice FAISS (inner product su vettori normalizzati per coseno)
        emb_array = np.array(embeddings, dtype=np.float32)
        if emb_array.ndim != 2 or emb_array.shape[0] == 0:
            raise ValueError("Embeddings non validi per costruzione FAISS")
        dim = emb_array.shape[1]
        faiss.normalize_L2(emb_array)
        index = faiss.IndexFlatIP(dim)
        index.add(emb_array)

        # 3) Salva indice FAISS
        faiss.write_index(index, str(faiss_path))

        # 4) Salva metadati + chunk originali in pickle
        chunk_records = []
        for i, c in enumerate(chunks):
            chunk_records.append({
                "chunk_id": c.chunk_id,
                "content": c.content,
                "metadata": c.metadata,
                "start_index": c.start_index,
                "end_index": c.end_index,
                "embedding_index": i
            })

        payload = {
            "document": {
                "source_path": doc.source_path,
                "output_path": getattr(doc, "output_path", None),
                "title": doc.metadata.get("title") if getattr(doc, "metadata", None) else None,
            },
            "chunks": chunk_records,
            "faiss_index_file": str(faiss_path)
        }

        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)
        
        self.logger.info(f"Artefatti chunk salvati: {faiss_path.name}, {pkl_path.name}")
    
    def _load_chunked_artifacts(self, doc: ProcessedDocument) -> Optional[List[SemanticChunk]]:
        """Se presenti, carica i chunk da chunked_document/ evitando di rifare il chunking.
        Restituisce None se i file non esistono o in caso di errore di caricamento.
        """
        try:
            base_dir = Path("chunked_document")
            try:
                stem = Path(doc.output_path).stem if getattr(doc, "output_path", None) else Path(doc.source_path).stem
            except Exception:
                stem = Path(doc.source_path).stem

            pkl_path = base_dir / f"{stem}.pkl"
            # La presenza del pickle è sufficiente per ricostruire i chunk originali
            if not pkl_path.exists():
                return None

            with open(pkl_path, "rb") as f:
                payload = pickle.load(f)

            chunk_records = payload.get("chunks", [])
            if not chunk_records:
                return None

            loaded_chunks: List[SemanticChunk] = []
            for rec in chunk_records:
                loaded_chunks.append(SemanticChunk(
                    content=rec.get("content", ""),
                    metadata=rec.get("metadata", {}),
                    chunk_id=rec.get("chunk_id", ""),
                    start_index=rec.get("start_index", 0),
                    end_index=rec.get("end_index", 0),
                ))

            return loaded_chunks
        except Exception as e:
            self.logger.warning(f"Errore nel caricare chunk precomputati: {e}")
            return None

    # def _load_cache(self) -> bool:
    #     """Carica lo stato dalla cache"""
    #     try:
    #         if self.cache_file.exists():
    #             with open(self.cache_file, 'rb') as f:
    #                 self.state = pickle.load(f)
                
    #             # Ricostruisci gli indici
    #             if self.state.semantic_chunks:
    #                 self.retriever.build_indices(self.state.semantic_chunks)
    #                 self.state.indices_built = True
                
    #             self.logger.info("Cache caricata con successo")
    #             return True
    #     except Exception as e:
    #         self.logger.warning(f"Errore nel caricare la cache: {e}")
        
    #     return False
    
    # def _save_cache(self):
    #     """Salva lo stato nella cache"""
    #     try:
    #         self.cache_file.parent.mkdir(parents=True, exist_ok=True)
    #         with open(self.cache_file, 'wb') as f:
    #             pickle.dump(self.state, f)
    #         self.logger.info("Cache salvata con successo")
    #     except Exception as e:
    #         self.logger.warning(f"Errore nel salvare la cache: {e}")
    
    # def clear_cache(self):
    #     """Pulisce la cache"""
    #     try:
    #         if self.cache_file.exists():
    #             self.cache_file.unlink()
    #         self.state = None
    #         self.logger.info("Cache pulita")
    #     except Exception as e:
    #         self.logger.error(f"Errore nel pulire la cache: {e}")


def create_rag_pipeline(config: Optional[RAGConfig] = None) -> AdvancedRAGPipeline:
    """Factory function per creare una pipeline RAG"""
    return AdvancedRAGPipeline(config)
