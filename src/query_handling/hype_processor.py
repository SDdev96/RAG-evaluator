"""
HyPE (Hypothetical Prompt Embeddings) per il query handling avanzato
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage

from config.config import HyPEConfig
from src.chunking.semantic_chunker import SemanticChunk


@dataclass
class HypotheticalQuestion:
    """Rappresenta una domanda ipotetica generata per un chunk"""
    question: str
    chunk_id: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class EnrichedChunk:
    """Chunk arricchito con domande ipotetiche"""
    original_chunk: SemanticChunk
    hypothetical_questions: List[HypotheticalQuestion]
    embeddings: List[List[float]]  # Embeddings delle domande ipotetiche


class HyPEProcessor:
    """
    Processore HyPE che genera domande ipotetiche per ogni chunk
    e crea embeddings ottimizzati per il retrieval
    """
    
    def __init__(self, config: HyPEConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Inizializza il modello di linguaggio
        self.llm = ChatOpenAI(
            model=config.language_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Inizializza gli embeddings
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model
        )
        
        self.logger.info(f"HyPE Processor inizializzato con {config.language_model}")
    
    def generate_hypothetical_questions(self, chunk: SemanticChunk) -> List[HypotheticalQuestion]:
        """
        Genera domande ipotetiche per un singolo chunk
        
        Args:
            chunk: Chunk per cui generare le domande
            
        Returns:
            List[HypotheticalQuestion]: Lista di domande ipotetiche
        """
        try:
            # Crea il prompt per generare domande ipotetiche
            system_prompt = """Sei un esperto nell'analisi di documenti tecnici. 
            Il tuo compito è generare domande ipotetiche che un utente potrebbe fare 
            per cercare le informazioni contenute nel testo fornito.
            
            Regole:
            1. Genera esattamente {num_questions} domande diverse
            2. Le domande devono essere specifiche e pertinenti al contenuto
            3. Usa un linguaggio naturale e vario
            4. Includi sia domande dirette che concettuali
            5. Ogni domanda deve essere su una riga separata
            6. Non numerare le domande
            
            Esempio di formato:
            Come si configura il sistema di autenticazione?
            Quali sono i requisiti hardware minimi?
            Che cosa succede in caso di errore di connessione?"""
            
            human_prompt = f"""Contenuto del documento:
            
            {chunk.content[:2000]}...
            
            Genera {self.config.num_hypothetical_questions} domande ipotetiche che un utente potrebbe fare per trovare queste informazioni."""
            
            # Genera le domande
            messages = [
                SystemMessage(content=system_prompt.format(num_questions=self.config.num_hypothetical_questions)),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            questions_text = response.content.strip()
            
            # Processa le domande generate
            questions = []
            for i, question_line in enumerate(questions_text.split('\n')):
                question = question_line.strip()
                if question and not question.startswith('#'):
                    # Rimuove numerazione se presente
                    question = self._clean_question(question)
                    
                    hyp_question = HypotheticalQuestion(
                        question=question,
                        chunk_id=chunk.chunk_id,
                        confidence=1.0 - (i * 0.1),  # Confidence decrescente
                        metadata={
                            "generation_method": "llm",
                            "source_chunk_size": len(chunk.content),
                            "question_index": i
                        }
                    )
                    questions.append(hyp_question)
            
            self.logger.debug(f"Generate {len(questions)} domande per chunk {chunk.chunk_id}")
            return questions
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione di domande per chunk {chunk.chunk_id}: {e}")
            # Genera domande di fallback
            return self._generate_fallback_questions(chunk)
    
    def _clean_question(self, question: str) -> str:
        """Pulisce una domanda rimuovendo numerazione e caratteri indesiderati"""
        # Rimuove numerazione all'inizio
        import re
        question = re.sub(r'^\d+[\.\)]\s*', '', question)
        question = re.sub(r'^[-\*]\s*', '', question)
        
        # Rimuove spazi extra
        question = ' '.join(question.split())
        
        # Assicura che termini con punto interrogativo
        if not question.endswith('?'):
            question += '?'
        
        return question
    
    def _generate_fallback_questions(self, chunk: SemanticChunk) -> List[HypotheticalQuestion]:
        """Genera domande di fallback quando la generazione LLM fallisce"""
        fallback_questions = [
            f"Che cosa dice il documento riguardo a {chunk.chunk_id}?",
            f"Quali informazioni sono contenute in questa sezione?",
            f"Come funziona il processo descritto qui?"
        ]
        
        questions = []
        for i, question in enumerate(fallback_questions[:self.config.num_hypothetical_questions]):
            hyp_question = HypotheticalQuestion(
                question=question,
                chunk_id=chunk.chunk_id,
                confidence=0.5,
                metadata={
                    "generation_method": "fallback",
                    "question_index": i
                }
            )
            questions.append(hyp_question)
        
        return questions
    
    def process_chunks(self, chunks: List[SemanticChunk], 
                      use_parallel: bool = True) -> List[EnrichedChunk]:
        """
        Processa una lista di chunks generando domande ipotetiche ed embeddings
        
        Args:
            chunks: Lista di chunks da processare
            use_parallel: Se usare il processing parallelo
            
        Returns:
            List[EnrichedChunk]: Lista di chunks arricchiti
        """
        self.logger.info(f"Processando {len(chunks)} chunks con HyPE")
        
        enriched_chunks = []
        
        if use_parallel and len(chunks) > 1:
            # Processing parallelo
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_chunk = {
                    executor.submit(self._process_single_chunk, chunk): chunk 
                    for chunk in chunks
                }
                
                for future in tqdm(as_completed(future_to_chunk), 
                                 total=len(chunks), 
                                 desc="Processing chunks"):
                    chunk = future_to_chunk[future]
                    try:
                        enriched_chunk = future.result()
                        enriched_chunks.append(enriched_chunk)
                    except Exception as e:
                        self.logger.error(f"Errore nel processare chunk {chunk.chunk_id}: {e}")
        else:
            # Processing sequenziale
            for chunk in tqdm(chunks, desc="Processing chunks"):
                try:
                    enriched_chunk = self._process_single_chunk(chunk)
                    enriched_chunks.append(enriched_chunk)
                except Exception as e:
                    self.logger.error(f"Errore nel processare chunk {chunk.chunk_id}: {e}")
        
        self.logger.info(f"Processati {len(enriched_chunks)} chunks con successo")
        return enriched_chunks
    
    def _process_single_chunk(self, chunk: SemanticChunk) -> EnrichedChunk:
        """Processa un singolo chunk"""
        # Genera domande ipotetiche
        hypothetical_questions = self.generate_hypothetical_questions(chunk)
        
        # Genera embeddings per le domande
        question_texts = [q.question for q in hypothetical_questions]
        embeddings_list = []
        
        if question_texts:
            try:
                embeddings_result = self.embeddings.embed_documents(question_texts)
                embeddings_list = embeddings_result
            except Exception as e:
                self.logger.error(f"Errore nella generazione embeddings per chunk {chunk.chunk_id}: {e}")
                # Embeddings vuoti come fallback
                embeddings_list = [[0.0] * 1536] * len(question_texts)
        
        return EnrichedChunk(
            original_chunk=chunk,
            hypothetical_questions=hypothetical_questions,
            embeddings=embeddings_list
        )
    
    def process_query(self, query: str) -> List[float]:
        """
        Processa una query utente creando il suo embedding
        
        Args:
            query: Query dell'utente
            
        Returns:
            List[float]: Embedding della query
        """
        try:
            query_embedding = self.embeddings.embed_query(query)
            return query_embedding
        except Exception as e:
            self.logger.error(f"Errore nell'embedding della query: {e}")
            return [0.0] * 1536  # Embedding vuoto come fallback
    
    def get_processing_statistics(self, enriched_chunks: List[EnrichedChunk]) -> Dict[str, Any]:
        """Calcola statistiche sul processing HyPE"""
        if not enriched_chunks:
            return {}
        
        total_questions = sum(len(chunk.hypothetical_questions) for chunk in enriched_chunks)
        avg_questions_per_chunk = total_questions / len(enriched_chunks)
        
        generation_methods = []
        for chunk in enriched_chunks:
            for question in chunk.hypothetical_questions:
                generation_methods.append(question.metadata.get("generation_method", "unknown"))
        
        method_counts = {}
        for method in generation_methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        stats = {
            "total_chunks_processed": len(enriched_chunks),
            "total_hypothetical_questions": total_questions,
            "avg_questions_per_chunk": avg_questions_per_chunk,
            "generation_method_distribution": method_counts,
            "chunks_with_embeddings": sum(1 for chunk in enriched_chunks if chunk.embeddings)
        }
        
        return stats


def create_hype_processor(config: Optional[HyPEConfig] = None) -> HyPEProcessor:
    """Factory function per creare un processore HyPE"""
    if config is None:
        config = HyPEConfig()
    
    return HyPEProcessor(config)
