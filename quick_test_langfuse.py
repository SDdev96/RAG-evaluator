"""
Quick test per Langfuse con Gemini e funzionalità avanzate di tracing.

Questo script testa:
1. Inizializzazione Langfuse con Gemini
2. Traces personalizzate
3. Spans annidati
4. Scores custom
5. Generazioni multiple
6. Metriche e valutazioni

Usage:
    python quick_test_langfuse.py
"""

import os
import time
import uuid
import random
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv

# Langfuse imports (v2/v3 compatible)
from langfuse import observe

# Try to import langfuse_context for v2 compatibility
try:
    from langfuse.decorators import langfuse_context
    LANGFUSE_V2 = True
except ImportError:
    # v3 doesn't have langfuse_context in decorators
    LANGFUSE_V2 = False
    langfuse_context = None

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Import nostro manager
from src.telemetry.langfuse_setup_test import LangfuseManager, LangfuseConfig

# Load environment variables
load_dotenv()

class LangfuseQuickTest:
    """
    Classe per testare tutte le funzionalità di Langfuse con Gemini.
    """
    
    def __init__(self):
        """Inizializza il test con Langfuse e Gemini."""
        print("🚀 Inizializzazione Quick Test Langfuse...")
        
        # Setup Langfuse
        self.langfuse_config = LangfuseConfig.from_environment()
        self.langfuse_manager = LangfuseManager(self.langfuse_config)
        
        # Setup Gemini
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("❌ GOOGLE_API_KEY non trovata nelle variabili d'ambiente")
        
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=self.google_api_key,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Direct Langfuse client per operazioni avanzate
        self.langfuse = self.langfuse_manager.get_client()
        
        print("✅ Inizializzazione completata!")
    
    @observe(name="simple_generation_test", as_type="generation")
    def test_simple_generation(self) -> str:
        """Test base: generazione semplice con trace automatica."""
        print("\n📝 Test 1: Generazione Semplice")
        
        prompt = "Dimmi una curiosità interessante sulla fisica quantistica in 2 frasi."
        
        # Usa il nostro manager per invocare il modello
        response = self.langfuse_manager.invoke_model_with_langchain(
            self.model,
            [HumanMessage(content=prompt)],
            extra_config={"temperature": 0.3}
        )
        
        result = response.content if hasattr(response, 'content') else str(response)
        print(f"🔬 Risposta: {result[:100]}...")
        return result

    def generate_and_trace(self) -> Any:
        """
        Funzione semplice che:
        1. Chiama il LLM con model.invoke()
        2. Crea una trace su Langfuse con i dettagli della chiamata
        
        Args:
            langfuse_manager: Istanza del LangfuseManager
            model: Modello LangChain
            prompt: Prompt da inviare
            config: Configurazione per model.invoke()
        
        Returns:
            Response dal modello
        """
        
        prompt = "Dimmi una curiosità interessante sulla fisica quantistica in 2 frasi."

        config = {}
        
        # 1. Chiama il LLM
        response = self.model.invoke(prompt, config=config)

        # Estrai numero di tokens
        input_tokens = response.usage_metadata.get('input_tokens', 0)
        output_tokens = response.usage_metadata.get('output_tokens', 0)
        total_tokens = input_tokens + output_tokens
        
        # Estrai contenuto della risposta
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # 2. Crea trace su Langfuse
        # langfuse_client = self.langfuse
        
        if self.langfuse:
            try:
                # Crea la trace con tutti i dettagli della chiamata LLM
                with self.langfuse.start_as_current_span(
                    name = "test_trace",
                    input = {
                        "prompt": prompt,
                        "config": config
                    },
                    output = response_content,
                    version = "test_version",
                    level="test_level",
                    status_message="test_status_message",
                    
                ) as root_span:

                    # Aggiunge gli attributi del primo span
                    root_span.update_trace(
                        user_id="test_user",
                        session_id="test_session",
                        version = "test_version",
                        tags=["test_tag"],
                        public = None
                    )

                    # Crea una generazione innestata (gen1)
                    with self.langfuse.start_as_current_observation(
                        name="test-gen1",
                        as_type='generation',
                        input={
                            "prompt": prompt,
                            "config": config
                        }
                    ) as gen1:
                        gen1.update(
                            output = response,
                            metadata = {
                                "confidence" : 0.9
                            },
                            version = None,
                            level = "DEFAULT",
                            status_message = "test_status_message",
                            completion_start_time = datetime.now().isoformat(),
                            model = self.model.model,
                            model_parameters = {"temperature": self.model.temperature},
                            usage_details = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens},
                            cost_details = {"input_tokens": 0.3, "output_tokens": 0.5, "total_tokens": 0.8},
                        )

                    # Crea uno span intermedio (mid_span)
                    with root_span.start_as_current_observation(name="second_span", as_type='span') as second_span:

                        # Aggiunge gli attributi del primo span
                        second_span.update(
                            input = "second span input",
                            output = "second span output"
                        )

                        # Crea una generazione innestata (gen2)
                        with second_span.start_as_current_observation(name="gen2", as_type='generation') as gen2:
                            gen2.update(output = response)

                        
            except Exception as e:
                print(f"Errore creazione trace: {e}")
        
        return response

    @observe(name="complex_rag_simulation", as_type="generation")
    def test_complex_rag_simulation(self) -> Dict[str, Any]:
        """Test complesso: simulazione RAG con spans multipli."""
        print("\n🔍 Test 2: Simulazione RAG Complessa")
        
        user_query = "Come funziona il machine learning nel riconoscimento delle immagini?"
        
        # Simula il retrieval di documenti
        retrieved_docs = self._simulate_document_retrieval(user_query)
        
        # Simula il reranking
        reranked_docs = self._simulate_reranking(retrieved_docs, user_query)
        
        # Genera la risposta finale
        final_response = self._generate_final_response(user_query, reranked_docs)
        
        # Aggiungi scores personalizzati
        self._add_custom_scores(user_query, final_response)
        
        return {
            "query": user_query,
            "retrieved_docs": len(retrieved_docs),
            "final_response": final_response,
            "processing_time": time.time()
        }
    
    @observe(name="document_retrieval", as_type="span")
    def _simulate_document_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Simula il retrieval di documenti."""
        print("  📚 Simulando retrieval documenti...")
        
        # Simula latenza
        time.sleep(0.1)
        
        # Documenti fittizi
        docs = [
            {
                "id": f"doc_{i}",
                "content": f"Documento {i} sul machine learning e computer vision...",
                "score": random.uniform(0.6, 0.95),
                "source": f"source_{i}.pdf"
            }
            for i in range(5)
        ]
        
        # Log come evento Langfuse (v2/v3 compatible)
        if LANGFUSE_V2 and langfuse_context:
            langfuse_context.update_current_observation(
                input={"query": query},
                output={"documents_found": len(docs)},
                metadata={
                    "retrieval_method": "semantic_search",
                    "embedding_model": "text-embedding-ada-002"
                }
            )
        # Note: v3 uses different context management through OTEL
        
        return docs
    
    @observe(name="document_reranking", as_type="span")
    def _simulate_reranking(self, docs: List[Dict], query: str) -> List[Dict]:
        """Simula il reranking dei documenti."""
        print("  🔄 Simulando reranking...")
        
        time.sleep(0.05)
        
        # Simula reranking (ordina per score decrescente)
        reranked = sorted(docs, key=lambda x: x["score"], reverse=True)[:3]
        
        if LANGFUSE_V2 and langfuse_context:
            langfuse_context.update_current_observation(
                input={"documents": len(docs), "query": query},
                output={"reranked_documents": len(reranked)},
                metadata={
                    "reranking_model": "cross-encoder",
                    "top_k": 3
                }
            )
        
        return reranked
    
    @observe(name="response_generation", as_type="generation")
    def _generate_final_response(self, query: str, context_docs: List[Dict]) -> str:
        """Genera la risposta finale usando il contesto."""
        print("  🤖 Generando risposta finale...")
        
        # Costruisci il prompt con contesto
        context = "\n".join([doc["content"] for doc in context_docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Sei un assistente AI esperto. Usa il contesto fornito per rispondere alla domanda."),
            ("human", f"Contesto:\n{context}\n\nDomanda: {query}")
        ])
        
        # Genera risposta
        response = self.langfuse_manager.invoke_model_with_langchain(
            self.model,
            prompt.format_messages(),
            extra_config={
                "temperature": 0.3,
                "max_tokens": 500
            }
        )
        
        result = response.content if hasattr(response, 'content') else str(response)
        print(f"  💬 Risposta generata: {result[:80]}...")
        
        return result
    
    def _add_custom_scores(self, query: str, response: str) -> None:
        """Aggiunge scores personalizzati alla trace corrente."""
        print("  📊 Aggiungendo scores personalizzati...")
        
        # Simula diverse metriche
        scores = {
            "relevance": random.uniform(0.7, 1.0),
            "accuracy": random.uniform(0.6, 0.95),
            "completeness": random.uniform(0.65, 0.9),
            "clarity": random.uniform(0.75, 1.0),
            "response_length": len(response.split()),
        }
        
        # Aggiungi scores tramite il client Langfuse diretto
        if LANGFUSE_V2 and langfuse_context:
            trace_id = langfuse_context.get_current_trace_id()
        else:
            # v3 usa OTEL context, più complesso - per ora usiamo il client diretto
            trace_id = None
        
        for score_name, score_value in scores.items():
            try:
                if trace_id:
                    self.langfuse.score(
                        trace_id=trace_id,
                        name=score_name,
                        value=score_value,
                        comment=f"Automated evaluation for {score_name}"
                    )
                    print(f"    ✅ Score '{score_name}': {score_value:.3f}")
                else:
                    print(f"    ⚠️  Score '{score_name}': {score_value:.3f} (trace_id not available in v3)")
            except Exception as e:
                print(f"    ❌ Errore aggiungendo score {score_name}: {e}")
    
    def test_batch_processing(self) -> List[Dict[str, Any]]:
        """Test batch: processa multiple queries con tracing."""
        print("\n📦 Test 3: Batch Processing")
        
        queries = [
            "Cos'è l'intelligenza artificiale?",
            "Come funziona una rete neurale?",
            "Quali sono le applicazioni del deep learning?",
            "Che differenza c'è tra ML e AI?"
        ]
        
        results = []
        
        with self.langfuse_manager:
            for i, query in enumerate(queries, 1):
                print(f"  🔄 Processing query {i}/{len(queries)}")
                
                # Crea una trace personalizzata per ogni query
                trace = self.langfuse.trace(
                    name=f"batch_query_{i}",
                    input={"query": query, "batch_position": i},
                    metadata={
                        "batch_id": str(uuid.uuid4()),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                try:
                    # Genera risposta
                    response = self.langfuse_manager.invoke_model_with_langchain(
                        self.model,
                        [HumanMessage(content=query)],
                        extra_config={"temperature": 0.6}
                    )
                    
                    result_text = response.content if hasattr(response, 'content') else str(response)
                    
                    # Aggiorna la trace con il risultato
                    trace.update(
                        output={"response": result_text},
                        metadata={
                            "response_length": len(result_text),
                            "processing_successful": True
                        }
                    )
                    
                    # Aggiungi score di qualità
                    quality_score = random.uniform(0.7, 0.95)
                    trace.score(
                        name="batch_quality",
                        value=quality_score,
                        comment="Automated batch processing quality score"
                    )
                    
                    results.append({
                        "query": query,
                        "response": result_text[:100] + "...",
                        "quality_score": quality_score,
                        "trace_id": trace.id
                    })
                    
                except Exception as e:
                    print(f"    ❌ Errore processing query {i}: {e}")
                    trace.update(
                        output={"error": str(e)},
                        metadata={"processing_successful": False}
                    )
                    
                    results.append({
                        "query": query,
                        "response": f"Error: {str(e)}",
                        "quality_score": 0.0,
                        "trace_id": trace.id
                    })
        
        return results
    
    @observe(name="conversation_simulation", as_type="generation")
    def test_conversation_flow(self) -> List[Dict[str, Any]]:
        """Test conversazione: simula un dialogo multi-turn."""
        print("\n💬 Test 4: Conversazione Multi-turn")
        
        conversation_history = []
        topics = [
            "Spiegami cos'è il quantum computing",
            "Come si collega alla crittografia?",
            "Quali sono i vantaggi rispetto ai computer classici?",
            "Quando sarà disponibile commercialmente?"
        ]
        
        for turn, user_message in enumerate(topics, 1):
            print(f"  🗣️  Turn {turn}: {user_message}")
            
            # Costruisci il contesto della conversazione
            messages = []
            
            # Aggiungi la storia della conversazione
            for hist in conversation_history[-3:]:  # Ultimi 3 scambi
                messages.append(HumanMessage(content=hist["user"]))
                messages.append(AIMessage(content=hist["assistant"]))
            
            # Aggiungi il messaggio corrente
            messages.append(HumanMessage(content=user_message))
            
            # Genera risposta con contesto
            response = self.langfuse_manager.invoke_model_with_langchain(
                self.model,
                messages,
                extra_config={
                    "temperature": 0.4,
                    "max_tokens": 300
                }
            )
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Salva nella storia
            conversation_history.append({
                "turn": turn,
                "user": user_message,
                "assistant": response_text,
                "timestamp": datetime.now().isoformat()
            })
            
            # Aggiungi scores di conversazione
            self._add_conversation_scores(turn, user_message, response_text)
            
            print(f"    🤖 Risposta: {response_text[:60]}...")
        
        return conversation_history
    
    def _add_conversation_scores(self, turn: int, user_msg: str, assistant_msg: str) -> None:
        """Aggiunge scores specifici per la conversazione."""
        if LANGFUSE_V2 and langfuse_context:
            trace_id = langfuse_context.get_current_trace_id()
        else:
            trace_id = None
        
        scores = {
            "conversation_coherence": min(1.0, 0.6 + (turn * 0.1)),  # Migliora nel tempo
            "context_retention": random.uniform(0.7, 0.9),
            "response_relevance": random.uniform(0.8, 1.0),
        }
        
        for score_name, score_value in scores.items():
            if trace_id:
                self.langfuse.score(
                    trace_id=trace_id,
                    name=score_name,
                    value=score_value,
                    comment=f"Turn {turn} - {score_name} evaluation"
                )
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test gestione errori con tracing."""
        print("\n⚠️  Test 5: Gestione Errori")
        
        error_scenarios = [
            {"prompt": "A" * 50000, "expected_error": "Token limit"},  # Prompt troppo lungo
            {"prompt": "", "expected_error": "Empty prompt"},  # Prompt vuoto
            {"prompt": "Normal prompt", "config": {"temperature": 5.0}, "expected_error": "Invalid temperature"}  # Config invalida
        ]
        
        results = []
        
        for i, scenario in enumerate(error_scenarios, 1):
            print(f"  🧪 Scenario errore {i}: {scenario['expected_error']}")
            
            # Crea trace per l'errore
            trace = self.langfuse.trace(
                name=f"error_scenario_{i}",
                input={"scenario": scenario["expected_error"]},
                metadata={"test_type": "error_handling"}
            )
            
            try:
                config = scenario.get("config", {"temperature": 0.7})
                response = self.langfuse_manager.invoke_model(
                    self.model,
                    [HumanMessage(content=scenario["prompt"])],
                    extra_config=config
                )
                
                # Se non c'è errore, segna come inaspettato
                trace.update(
                    output={"unexpected_success": True},
                    level="WARNING"
                )
                
                results.append({
                    "scenario": scenario["expected_error"],
                    "result": "Unexpected success",
                    "trace_id": trace.id
                })
                
            except Exception as e:
                # Errore atteso
                trace.update(
                    output={"error": str(e)},
                    level="ERROR",
                    status_message=f"Expected error: {str(e)[:100]}"
                )
                
                # Aggiungi score di error handling
                trace.score(
                    name="error_handling_success",
                    value=1.0,
                    comment="Successfully caught and handled expected error"
                )
                
                results.append({
                    "scenario": scenario["expected_error"],
                    "result": f"Caught: {str(e)[:50]}...",
                    "trace_id": trace.id
                })
                
                print(f"    ✅ Errore gestito correttamente: {str(e)[:50]}...")
        
        return {"error_scenarios": results}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Esegue tutti i test e fornisce un summary."""
        print("=" * 60)
        print("🎯 QUICK TEST LANGFUSE - INIZIO")
        print("=" * 60)
        
        start_time = time.time()
        all_results = {}
        
        try:
            # Test 1: Generazione semplice
            # all_results["simple_generation"] = self.test_simple_generation()

            all_results["generate_and_trace"] = self.generate_and_trace()
            
            # Test 2: RAG simulation
            # all_results["rag_simulation"] = self.test_complex_rag_simulation()
            
            # Test 3: Batch processing
            # all_results["batch_processing"] = self.test_batch_processing()
            
            # Test 4: Conversazione
            # all_results["conversation"] = self.test_conversation_flow()
            
            # Test 5: Error handling
            # all_results["error_handling"] = self.test_error_handling()
            
            execution_time = time.time() - start_time
            
            # Summary finale
            print("\n" + "=" * 60)
            print("📊 SUMMARY FINALE")
            print("=" * 60)
            print(f"⏱️  Tempo totale esecuzione: {execution_time:.2f}s")
            print(f"🔗 Traces create: ~{len(all_results) * 3}")  # Stima approssimativa
            print(f"📈 Scores aggiunti: ~{len(all_results) * 5}")  # Stima approssimativa
            print("✅ Tutti i test completati con successo!")
            
            # Flush finale per assicurarsi che tutto sia inviato
            if self.langfuse:
                self.langfuse.flush()
                print("🚀 Dati inviati a Langfuse!")
            
            return {
                "results": all_results,
                "execution_time": execution_time,
                "status": "success"
            }
            
        except Exception as e:
            print(f"\n❌ ERRORE DURANTE I TEST: {e}")
            return {
                "results": all_results,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error": str(e)
            }
        
        finally:
            # Cleanup
            self.langfuse_manager.cleanup()
            print("🧹 Cleanup completato")


def main():
    """Funzione principale per eseguire i test."""
    try:
        tester = LangfuseQuickTest()
        results = tester.run_all_tests()
        
        print("\n🎉 Quick Test completato!")
        print(f"Status: {results['status']}")
        
        if results['status'] == 'error':
            print(f"Errore: {results.get('error', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Errore fatale: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()