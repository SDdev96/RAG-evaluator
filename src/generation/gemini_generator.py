"""
Generatore di risposte usando l'API Google Gemini
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from src.telemetry.langfuse_setup import init_langfuse, invoke_with_langfuse

from config.config import GenerationConfig
from src.retrieval.fusion_retriever import RetrievalResult
from src.utils.helpers import compute_token_costs


@dataclass
class GenerationResult:
    """Risultato della generazione"""
    answer: str
    sources_used: List[str]
    confidence: float
    metadata: List[Dict[str, Any]]
    generation_stats: Dict[str, Any]


class GeminiGenerator:
    """
    Generatore di risposte usando Google Gemini
    """
    
    def __init__(self, config: GenerationConfig, api_key: str):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Inizializza il modello tramite LangChain wrapper
        # GOOGLE_API_KEY dev'essere presente nell'ambiente o passato internamente dal pacchetto
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
        )

        # Prova a inizializzare Langfuse e invocare con callback per tracciare input/output
        self.lf_client, self.lf_handler = init_langfuse()

        self.logger.info(f"Gemini Generator inizializzato con modello {config.model_name}")
    
    def generate_answer(self, query: str, retrieved_chunks: List[RetrievalResult], 
                       context: Optional[Dict[str, Any]] = None) -> GenerationResult:
        """
        Genera una risposta basata sulla query e sui chunks recuperati
        
        Args:
            query: Domanda dell'utente
            retrieved_chunks: Chunks recuperati dal retrieval
            context: Contesto aggiuntivo opzionale
            
        Returns:
            GenerationResult: Risultato della generazione
        """
        if not retrieved_chunks:
            return self._generate_no_context_answer(query)
        
        self.logger.info(f"Generando risposta per query con {len(retrieved_chunks)} chunks")
        
        try:
            # Costruisci il prompt
            prompt = self._build_prompt(query, retrieved_chunks, context)

            if self.lf_handler is not None:
                extra_meta = {
                    "component": "GeminiGenerator.generate_answer",
                    "num_chunks": len(retrieved_chunks),
                    "model": self.config.model_name,
                }
                lc_message = invoke_with_langfuse(
                    self.llm,
                    prompt,
                    self.lf_handler,
                    extra_config={"metadata": extra_meta},
                )
            else:
                # Fallback: invocazione senza Langfuse
                print("Langfuse non inizializzato, invocazione senza callback")
                self.logger.warning("Langfuse non inizializzato, invocazione senza callback")
                lc_message = self.llm.invoke(prompt)
            
            print("Risposta dell'LLM: ", lc_message)
            self.logger.info("Risposta dell'LLM: ", lc_message)

            # Accesso sicuro ai campi
            input_tokens = lc_message.usage_metadata.get('input_tokens', 0)  # Default a 0 se non presente
            output_tokens = lc_message.usage_metadata.get('output_tokens', 0)  # Default a 0 se non presente

            # Ora puoi usarli, ad esempio:
            token_cost = compute_token_costs(
                model_name=self.config.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                paid_level=False, # TODO: da modificare quando si passa al livello pagante
                prompt_length=len(prompt)
            )
            self.logger.info(f"Token cost: ${token_cost['total_cost_usd']}")
            print("Total cost: $", token_cost["total_cost_usd"])


            # Processa la risposta
            if getattr(lc_message, "content", None):
                answer = lc_message.content.strip()
                
                # Estrai le fonti utilizzate
                sources_used = [chunk.chunk_id for chunk in retrieved_chunks]
                
                # Calcola confidence basata sui scores dei chunks
                confidence = self._calculate_confidence(retrieved_chunks, lc_message)
                
                # Prepara metadati RAG
                rag_metadata = {
                    "model_used": self.config.model_name,
                    "num_sources": len(retrieved_chunks),
                    "query_length": len(query),
                    "answer_length": len(answer),
                    "prompt_length": len(prompt)
                }
                
                # Estrai metadati della risposta LLM
                response_metadata = getattr(lc_message, "response_metadata", {})
                
                # Estrai metadati di utilizzo
                usage_metadata = getattr(lc_message, "usage_metadata", {})
                
                # Crea la lista di metadati strutturata
                metadata = [
                    rag_metadata,      # RAG_metadata
                    response_metadata, # response_metadata
                    usage_metadata     # usage_metadata
                ]
                
                # Statistiche di generazione
                generation_stats = {
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                }
                
                result = GenerationResult(
                    answer=answer,
                    sources_used=sources_used,
                    confidence=confidence,
                    metadata=metadata,
                    generation_stats=generation_stats
                )
                
                self.logger.info("Risposta generata con successo")
                return result
            
            else:
                self.logger.warning("Risposta vuota da Gemini")
                return self._generate_fallback_answer(query, retrieved_chunks)
                
        except Exception as e:
            self.logger.error(f"Errore nella generazione con Gemini: {e}")
            return self._generate_fallback_answer(query, retrieved_chunks)
    
    def _build_prompt(self, query: str, chunks: List[RetrievalResult], 
                     context: Optional[Dict[str, Any]] = None) -> str:
        """Costruisce il prompt per Gemini"""
        
        # Prompt di sistema
        system_prompt = """Sei un assistente AI esperto nell'analisi di documenti tecnici. 
        Il tuo compito è rispondere alle domande degli utenti basandoti esclusivamente 
        sulle informazioni fornite nei documenti di riferimento.

        ISTRUZIONI IMPORTANTI:
        1. Rispondi SOLO basandoti sulle informazioni contenute nei documenti forniti
        2. Se le informazioni non sono sufficienti, dillo chiaramente
        3. Cita sempre le fonti quando possibile
        4. Usa un linguaggio chiaro e professionale
        5. Struttura la risposta in modo logico e comprensibile
        6. Se ci sono informazioni contraddittorie, evidenziale
        7. Non inventare informazioni non presenti nei documenti"""
        
        # Costruisci il contesto dai chunks
        context_text = "\n\n=== DOCUMENTI DI RIFERIMENTO ===\n"
        
        for i, chunk in enumerate(chunks, 1):
            context_text += f"\n--- Documento {i} (ID: {chunk.chunk_id}, Score: {chunk.score:.3f}) ---\n"
            context_text += chunk.content
            context_text += "\n"
        
        # Aggiungi contesto aggiuntivo se fornito
        if context:
            context_text += "\n=== CONTESTO AGGIUNTIVO ===\n"
            for key, value in context.items():
                context_text += f"{key}: {value}\n"
        
        # Costruisci il prompt finale
        prompt = f"""{system_prompt}

                    {context_text}

                    === DOMANDA DELL'UTENTE ===
                    {query}

                    === RISPOSTA ===
                    Basandoti esclusivamente sui documenti di riferimento forniti, rispondi alla domanda dell'utente in modo completo e accurato:"""
        
        return prompt
    
    def _calculate_confidence(self, chunks: List[RetrievalResult], response) -> float:
        """Calcola un punteggio di confidence per la risposta
        
        Args:
            chunks: Lista di RetrievalResult
            response: Risposta generata
        
        Returns:
            float: Punteggio di confidence
        """
        if not chunks:
            return 0.0
        
        # Base confidence sui scores dei chunks
        avg_chunk_score = sum(chunk.score for chunk in chunks) / len(chunks)
        
        # Fattore basato sul numero di chunks
        num_chunks_factor = min(len(chunks) / 3.0, 1.0)  # Massimo con 3+ chunks
        
        # Fattore basato sulla lunghezza della risposta (risposte troppo corte o lunghe sono sospette)
        response_length = len(getattr(response, "content", "")) if response else 0
        if 50 <= response_length <= 2000:
            length_factor = 1.0
        elif response_length < 50:
            length_factor = 0.5
        else:
            length_factor = 0.8
        
        # Combina i fattori
        confidence = avg_chunk_score * num_chunks_factor * length_factor
        
        return min(confidence, 1.0)
    
    def _extract_safety_ratings(self, response) -> Dict[str, Any]:
        """Estrae le valutazioni di sicurezza dalla risposta"""
        safety_ratings = {}
        
        # Non disponibile tramite LangChain wrapper: ritorna vuoto
        
        return safety_ratings

    def _generate_no_context_answer(self, query: str) -> GenerationResult:
        """Genera una risposta quando non ci sono chunks di contesto"""
        answer = """Mi dispiace, ma non ho trovato informazioni rilevanti nei documenti 
        disponibili per rispondere alla tua domanda. Potresti riformulare la domanda 
        o fornire più dettagli specifici?"""
        
        return GenerationResult(
            answer=answer,
            sources_used=[],
            confidence=0.0,
            metadata=[
                {
                    "model_used": self.config.model_name,
                    "num_sources": 0,
                    "query_length": len(query),
                    "answer_length": len(answer),
                    "generation_type": "no_context"
                },
                {},  # response_metadata vuoto
                {}   # usage_metadata vuoto
            ],
            generation_stats={}
        )
    
    def _generate_fallback_answer(self, query: str, chunks: List[RetrievalResult]) -> GenerationResult:
        """Genera una risposta di fallback in caso di errore"""
        answer = f"""Si è verificato un errore nella generazione della risposta. 
        Ho trovato {len(chunks)} documenti potenzialmente rilevanti, ma non sono 
        riuscito a elaborare una risposta completa. Ti suggerisco di riprovare 
        o riformulare la domanda."""
        
        sources_used = [chunk.chunk_id for chunk in chunks] if chunks else []
        
        return GenerationResult(
            answer=answer,
            sources_used=sources_used,
            confidence=0.1,
            metadata=[
                {
                    "model_used": self.config.model_name,
                    "num_sources": len(chunks),
                    "query_length": len(query),
                    "answer_length": len(answer),
                    "generation_type": "fallback"
                },
                {},  # response_metadata vuoto
                {}   # usage_metadata vuoto
            ],
            generation_stats={}
        )
    
    def generate_summary(self, chunks: List[RetrievalResult], 
                        summary_type: str = "comprehensive") -> GenerationResult:
        """
        Genera un riassunto dei chunks forniti
        
        Args:
            chunks: Chunks da riassumere
            summary_type: Tipo di riassunto ("comprehensive", "brief", "key_points")
            
        Returns:
            GenerationResult: Riassunto generato
        """
        if not chunks:
            return self._generate_no_context_answer("riassunto")
        
        # Percorso e nome file del riassunto (deterministico per tipo)
        try:
            summary_dir = Path("summary")
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_filename = f"summary_{summary_type}.md"
            summary_path = summary_dir / summary_filename
        except Exception as e:
            self.logger.warning(f"Impossibile preparare la cartella summary: {e}")
        
        # Prompt per il riassunto
        if summary_type == "brief":
            summary_instruction = "Crea un riassunto breve e conciso (massimo 200 parole)"
        elif summary_type == "key_points":
            summary_instruction = "Elenca i punti chiave più importanti in formato bullet point"
        else:  # comprehensive
            summary_instruction = "Crea un riassunto completo e dettagliato"
        
        context_text = "\n\n=== DOCUMENTI DA RIASSUMERE ===\n"
        for i, chunk in enumerate(chunks, 1):
            context_text += f"\n--- Sezione {i} ---\n{chunk.content}\n"
        
        # Se esiste già il riassunto salvato, non rigenerare
        try:
            if 'summary_path' in locals() and summary_path.exists():
                self.logger.info(f"Riassunto già presente: {summary_path}. Salto la generazione.")
                cached_answer = summary_path.read_text(encoding='utf-8')
                sources_used = [chunk.chunk_id for chunk in chunks]
                return GenerationResult(
                    answer=cached_answer.strip(),
                    sources_used=sources_used,
                    confidence=0.8,
                    metadata=[
                        {
                            "model_used": self.config.model_name,
                            "num_sources": len(chunks),
                            "summary_type": summary_type,
                            "answer_length": len(cached_answer),
                            "generation_type": "summary",
                            "cache": True,
                            "summary_path": str(summary_path)
                        },
                        {},  # response_metadata vuoto
                        {}   # usage_metadata vuoto
                    ],
                    generation_stats={}
                )
        except Exception as e:
            self.logger.warning(f"Impossibile leggere riassunto esistente: {e}. Procedo con la generazione.")
        
        prompt = f"""Sei un esperto nell'analisi e sintesi di documenti tecnici. 
            {context_text}

            === ISTRUZIONI ===
            {summary_instruction} dei documenti forniti sopra. 
            Mantieni l'accuratezza delle informazioni e usa un linguaggio chiaro e professionale.

            === RIASSUNTO ==="""
        
        try:
            lc_message = self.llm.invoke(prompt)

            if getattr(lc_message, "content", None):
                answer = lc_message.content.strip()
                sources_used = [chunk.chunk_id for chunk in chunks]
                confidence = 0.8  # Confidence alta per i riassunti
                
                # Salva il riassunto in summary/
                try:
                    if 'summary_path' in locals():
                        summary_path.write_text(answer, encoding='utf-8')
                        self.logger.info(f"Riassunto salvato in: {summary_path}")
                except Exception as e:
                    self.logger.warning(f"Impossibile salvare il riassunto: {e}")
                
                return GenerationResult(
                    answer=answer,
                    sources_used=sources_used,
                    confidence=confidence,
                    metadata=[
                        {
                            "model_used": self.config.model_name,
                            "num_sources": len(chunks),
                            "summary_type": summary_type,
                            "answer_length": len(answer),
                            "generation_type": "summary",
                            "cache": False,
                            "summary_path": str(summary_path) if 'summary_path' in locals() else None
                        },
                        {},  # response_metadata vuoto
                        {}   # usage_metadata vuoto
                    ],
                    generation_stats={}
                )
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione del riassunto: {e}")
        
        return self._generate_fallback_answer("riassunto", chunks)


def create_gemini_generator(config: Optional[GenerationConfig] = None, 
                           api_key: Optional[str] = None) -> GeminiGenerator:
    """Factory function per creare un generatore Gemini"""
    if config is None:
        config = GenerationConfig()
    
    if api_key is None:
        import os
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key di Google non fornita")
    
    return GeminiGenerator(config, api_key)
