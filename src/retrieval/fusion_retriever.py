"""
Fusion Retrieval che combina ricerca vettoriale e BM25
"""
import logging
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi

from config.config import FusionRetrievalConfig
from src.chunking.semantic_chunker import SemanticChunk
from src.embeddings.provider import EmbeddingsProvider


@dataclass
class RetrievalResult:
    """Risultato del retrieval"""
    chunk_id: str
    content: str
    score: float
    vector_score: float
    bm25_score: float
    metadata: Dict[str, Any]
    rank: int


class FusionRetriever:
    """
    Retriever che combina ricerca vettoriale (FAISS) e ricerca per parole chiave (BM25)
    """
    
    def __init__(self, config: FusionRetrievalConfig, embeddings_provider: EmbeddingsProvider):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Inizializza componenti
        self.embeddings_provider = embeddings_provider
        self.vector_index = None
        self.bm25_index = None
        self.chunks_data = []
        self.chunk_id_to_index = {}
        self._bm25_tokenized_texts: Optional[List[List[str]]] = None
        
        self.logger.info("Fusion Retriever inizializzato")
    
    def build_indices(self, chunks: List[SemanticChunk]):
        """
        Costruisce gli indici vettoriale e BM25 dai chunks semantici
        
        Args:
            chunks: Lista di chunks semantici
        """
        self.logger.info(f"Costruendo indici per {len(chunks)} chunks")
        
        # Prepara i dati
        all_embeddings = []
        all_texts = []
        chunk_mappings = []
        
        for chunk_idx, chunk in enumerate(chunks):
            try:
                chunk_embedding = self.embeddings_provider.embed_query(chunk.content)
                all_embeddings.append(chunk_embedding)
                all_texts.append(chunk.content)

                chunk_mappings.append({
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk_idx,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "type": "original_chunk"
                })
            except Exception as e:
                self.logger.error(f"Errore nel processare chunk {chunk.chunk_id}: {e}")
                continue
        
        # Costruisci indice vettoriale FAISS
        self._build_vector_index(all_embeddings, chunk_mappings)
        
        # Costruisci indice BM25
        self._build_bm25_index(all_texts, chunk_mappings)
        
        self.logger.info(f"Indici costruiti con {len(all_embeddings)} vettori e {len(all_texts)} testi")
    
    def _build_vector_index(self, embeddings: List[List[float]], chunk_mappings: List[Dict]):
        """Costruisce l'indice vettoriale FAISS"""
        if not embeddings:
            raise ValueError("Nessun embedding fornito per costruire l'indice vettoriale")
        
        # Converte in array numpy
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Crea indice FAISS
        dimension = embeddings_array.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)  # Inner Product per similarità coseno
        
        # Normalizza gli embeddings per la similarità coseno
        faiss.normalize_L2(embeddings_array)
        
        # Aggiungi all'indice
        self.vector_index.add(embeddings_array)
        
        # Salva i mappings
        self.chunks_data = chunk_mappings
        self.chunk_id_to_index = {
            mapping["chunk_id"]: idx for idx, mapping in enumerate(chunk_mappings)
        }
        
        self.logger.info(f"Indice vettoriale FAISS creato con dimensione {dimension}")
    
    def _build_bm25_index(self, texts: List[str], chunk_mappings: List[Dict]):
        """Costruisce l'indice BM25"""
        # Tokenizza i testi
        tokenized_texts = [self._tokenize(text) for text in texts]
        
        # Crea indice BM25
        self.bm25_index = BM25Okapi(tokenized_texts)
        self._bm25_tokenized_texts = tokenized_texts
        
        self.logger.info("Indice BM25 creato")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenizza un testo per BM25"""
        import re
        # Semplice tokenizzazione
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Esegue il retrieval fusion combinando ricerca vettoriale e BM25
        
        Args:
            query: Query dell'utente
            top_k: Numero di risultati da restituire (default: config.top_k)
            
        Returns:
            List[RetrievalResult]: Risultati ordinati per score fusion
        """
        if top_k is None:
            top_k = self.config.top_k
        
        if self.vector_index is None or self.bm25_index is None:
            raise ValueError("Indici non costruiti. Chiamare build_indices() prima del retrieval.")
        
        self.logger.debug(f"Eseguendo retrieval per query: '{query[:50]}...'")
        
        # Ricerca vettoriale
        vector_results = self._vector_search(query, top_k * 2)  # Prendi più risultati per la fusion
        
        # Ricerca BM25
        bm25_results = self._bm25_search(query, top_k * 2)
        
        # Combina i risultati
        fusion_results = self._combine_results(vector_results, bm25_results, top_k)
        
        self.logger.debug(f"Restituiti {len(fusion_results)} risultati fusion")
        return fusion_results

    def retrieve_multi(self, queries: List[str], top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Esegue retrieval combinando molteplici query (query transformations).

        Per ciascuna query esegue ricerca vettoriale e BM25 e combina i risultati
        aggregando gli score (max per tipo e poi media pesata).

        Args:
            queries: Lista di query
            top_k: Numero di risultati da restituire (default: config.top_k)
            
        Returns:
            List[RetrievalResult]: Risultati ordinati per score fusion
        """
        if top_k is None:
            top_k = self.config.top_k

        if not queries:
            return []

        # Accumula risultati per indice
        vector_acc: Dict[int, float] = {}
        bm25_acc: Dict[int, float] = {}

        for q in queries:
            vec = self._vector_search(q, top_k)
            bm = self._bm25_search(q, top_k)

            # Normalizza singolarmente per query per rendere comparabili
            vec_scores = self._normalize_scores([s for _, s in vec])
            bm_scores = self._normalize_scores([s for _, s in bm])

            for (idx, _), ns in zip(vec, vec_scores):
                vector_acc[idx] = max(vector_acc.get(idx, 0.0), ns)
            for (idx, _), ns in zip(bm, bm_scores):
                bm25_acc[idx] = max(bm25_acc.get(idx, 0.0), ns)

        # Converti in liste
        vector_results = list(vector_acc.items())
        bm25_results = list(bm25_acc.items())

        # Riusa combinazione esistente
        return self._combine_results(vector_results, bm25_results, top_k)
    
    def _vector_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Esegue ricerca vettoriale"""
        try:
            # Genera embedding della query
            query_embedding = self.embeddings_provider.embed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Normalizza per similarità coseno
            faiss.normalize_L2(query_vector)
            
            # Cerca
            scores, indices = self.vector_index.search(query_vector, k)
            
            # Restituisce (indice, score)
            results = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]
            return results
            
        except Exception as e:
            self.logger.error(f"Errore nella ricerca vettoriale: {e}")
            return []
    
    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Esegue ricerca BM25"""
        try:
            # Tokenizza la query
            query_tokens = self._tokenize(query)
            
            # Ottieni scores BM25
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Ordina per score e prendi i top k
            scored_indices = [(i, score) for i, score in enumerate(scores)]
            scored_indices.sort(key=lambda x: x[1], reverse=True)
            
            return scored_indices[:k]
            
        except Exception as e:
            self.logger.error(f"Errore nella ricerca BM25: {e}")
            return []
    
    def _combine_results(self, vector_results: List[Tuple[int, float]], 
                        bm25_results: List[Tuple[int, float]], 
                        top_k: int) -> List[RetrievalResult]:
        """Combina i risultati di ricerca vettoriale e BM25"""
        
        # Normalizza gli scores
        vector_scores = self._normalize_scores([score for _, score in vector_results])
        bm25_scores = self._normalize_scores([score for _, score in bm25_results])
        
        # Crea dizionari per lookup rapido
        vector_dict = {idx: norm_score for (idx, _), norm_score in zip(vector_results, vector_scores)}
        bm25_dict = {idx: norm_score for (idx, _), norm_score in zip(bm25_results, bm25_scores)}
        
        # Combina tutti gli indici unici
        all_indices = set(vector_dict.keys()) | set(bm25_dict.keys())
        
        # Calcola scores fusion
        fusion_results = []
        for idx in all_indices:
            if idx >= len(self.chunks_data):
                continue
                
            vector_score = vector_dict.get(idx, 0.0)
            bm25_score = bm25_dict.get(idx, 0.0)
            
            # Score fusion pesato
            fusion_score = (self.config.vector_weight * vector_score + 
                           self.config.bm25_weight * bm25_score)
            
            chunk_data = self.chunks_data[idx]
            
            result = RetrievalResult(
                chunk_id=chunk_data["chunk_id"],
                content=chunk_data["content"],
                score=fusion_score,
                vector_score=vector_score,
                bm25_score=bm25_score,
                metadata=chunk_data["metadata"],
                rank=0  # Verrà impostato dopo l'ordinamento
            )
            
            fusion_results.append(result)
        
        # Ordina per score fusion e assegna rank
        fusion_results.sort(key=lambda x: x.score, reverse=True)
        
        # Rimuovi duplicati basati su chunk_id mantenendo il migliore
        seen_chunks = set()
        unique_results = []
        
        for result in fusion_results:
            if result.chunk_id not in seen_chunks:
                seen_chunks.add(result.chunk_id)
                result.rank = len(unique_results) + 1
                unique_results.append(result)
                
                if len(unique_results) >= top_k:
                    break
        
        return unique_results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalizza una lista di scores tra 0 e 1"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def get_retrieval_statistics(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Calcola statistiche sui risultati del retrieval"""
        if not results:
            return {}
        
        vector_scores = [r.vector_score for r in results]
        bm25_scores = [r.bm25_score for r in results]
        fusion_scores = [r.score for r in results]
        
        stats = {
            "total_results": len(results),
            "avg_vector_score": sum(vector_scores) / len(vector_scores),
            "avg_bm25_score": sum(bm25_scores) / len(bm25_scores),
            "avg_fusion_score": sum(fusion_scores) / len(fusion_scores),
            "max_fusion_score": max(fusion_scores),
            "min_fusion_score": min(fusion_scores),
            "vector_weight_used": self.config.vector_weight,
            "bm25_weight_used": self.config.bm25_weight
        }
        
        return stats

    def save_indices(self, directory: str | os.PathLike):
        """Salva su disco l'indice vettoriale FAISS, il corpus tokenizzato per BM25
        e i metadati dei chunk necessari al retrieval.

        Files salvati nella directory:
        - vector.faiss: indice FAISS
        - corpus_bm25.pkl: lista di testi tokenizzati per ricreare BM25
        - chunks_data.pkl: lista di mapping per chunk (id, contenuto, metadata)
        """
        try:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)

            if self.vector_index is None or self.bm25_index is None or not self.chunks_data:
                raise ValueError("Indici non inizializzati: impossibile salvarli")

            self.logger.info(f"Salvataggio indici di retrieval in '{dir_path}...'")
            print(f"Salvataggio indici di retrieval in '{dir_path}...'")

            # 1) FAISS
            faiss_path = dir_path / "vector.faiss"
            faiss.write_index(self.vector_index, str(faiss_path))

            # 2) Corpus tokenizzato per BM25 (ricreabile)
            if self._bm25_tokenized_texts is None:
                raise ValueError("Corpus BM25 non disponibile per il salvataggio")
            with open(dir_path / "corpus_bm25.pkl", "wb") as f:
                pickle.dump(self._bm25_tokenized_texts, f)
            self.logger.info(f"Corpus BM25 salvato in '{dir_path / "corpus_bm25.pkl"}'")
            print(f"Corpus BM25 salvato in '{dir_path / "corpus_bm25.pkl"}'")

            # 3) Chunk mappings
            with open(dir_path / "chunks_data.pkl", "wb") as f:
                pickle.dump(self.chunks_data, f)
            self.logger.info(f"Chunk mappings salvati in '{dir_path / "chunks_data.pkl"}'")
            print(f"Chunk mappings salvati in '{dir_path / "chunks_data.pkl"}'")

            self.logger.info(f"Indici di retrieval salvati in '{dir_path}'")
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio degli indici: {e}")
            raise

    def load_indices(self, directory: str | os.PathLike) -> bool:
        """Carica indici di retrieval da disco, se presenti e validi.

        Returns True se il caricamento ha successo, False altrimenti.
        """
        try:
            dir_path = Path(directory)
            faiss_path = dir_path / "vector.faiss"
            bm25_path = dir_path / "corpus_bm25.pkl"
            chunks_path = dir_path / "chunks_data.pkl"

            if not (faiss_path.exists() and bm25_path.exists() and chunks_path.exists()):
                return False

            self.logger.info(f"Caricamento indici di retrieval da '{dir_path}'...")
            print(f"Caricamento indici di retrieval da '{dir_path}'...")

            # 1) Carica FAISS
            self.vector_index = faiss.read_index(str(faiss_path))

            # 2) Carica corpus BM25 e ricrea indice
            with open(bm25_path, "rb") as f:
                tokenized_texts = pickle.load(f)
            self._bm25_tokenized_texts = tokenized_texts
            self.bm25_index = BM25Okapi(tokenized_texts)
            self.logger.info(f"Corpus BM25 caricato da '{bm25_path}'")
            print(f"Corpus BM25 caricato da '{bm25_path}'")

            # 3) Carica chunk mappings
            with open(chunks_path, "rb") as f:
                self.chunks_data = pickle.load(f)
            self.chunk_id_to_index = {
                mapping["chunk_id"]: idx for idx, mapping in enumerate(self.chunks_data)
            }
            self.logger.info(f"Chunk mappings caricati da '{chunks_path}'")
            print(f"Chunk mappings caricati da '{chunks_path}'")

            self.logger.info(f"Indici di retrieval caricati da '{dir_path}'")
            print(f"Indici di retrieval caricati da '{dir_path}'")
            return True
        except Exception as e:
            self.logger.warning(f"Errore nel caricamento degli indici: {e}")
            print(f"Errore nel caricamento degli indici: {e}")
            return False


def create_fusion_retriever(config: Optional[FusionRetrievalConfig] = None, 
                           embeddings_provider: Optional[EmbeddingsProvider] = None) -> FusionRetriever:
    """Factory function per creare un fusion retriever"""
    if config is None:
        config = FusionRetrievalConfig()
    if embeddings_provider is None:
        raise ValueError("EmbeddingsProvider è richiesto per creare FusionRetriever")
    return FusionRetriever(config, embeddings_provider)
