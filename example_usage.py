"""
Esempio di utilizzo del sistema RAG avanzato
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

from config.config import get_default_config
from src.pipeline.advanced_rag_pipeline import create_rag_pipeline
from src.utils.logging_config import setup_logging
from src.utils.helpers import validate_api_keys, load_environment_variables


def main():
    """Esempio principale di utilizzo del sistema RAG"""
    
    # 1. Setup del logging
    logger = setup_logging(
        log_level="INFO",
        log_file="rag_example.log",
        console_output=True
    )
    
    logger.info("=== AVVIO SISTEMA RAG AVANZATO ===")
    
    # 2. Carica configurazione
    config = get_default_config()
    
    # 3. Valida le API keys
    required_keys = ["OPENAI_API_KEY", "GOOGLE_API_KEY"]
    api_validation = validate_api_keys(required_keys)
    
    for key, is_valid in api_validation.items():
        if not is_valid:
            logger.error(f"API key mancante: {key}")
            print(f"❌ Configura la variabile d'ambiente {key}")
            return
        else:
            logger.info(f"✅ API key {key} configurata")
    
    # 4. Inizializza la pipeline RAG
    try:
        rag_pipeline = create_rag_pipeline(config)
        logger.info("Pipeline RAG inizializzata con successo")
    except Exception as e:
        logger.error(f"Errore nell'inizializzazione: {e}")
        return
    
    # 5. Processa i documenti
    document_paths = [
        "data"  # Processa tutti i documenti nella directory data
    ]
    
    try:
        logger.info("Inizio processing dei documenti...")
        processed_docs = rag_pipeline.process_documents(
            document_paths, 
            force_reprocess=False  # Usa cache se disponibile
        )
        logger.info(f"✅ Processati {len(processed_docs)} documenti")
        
    except Exception as e:
        logger.error(f"Errore nel processing: {e}")
        return
    
    # 6. Esempi di query
    example_queries = [
        "Come si configura il sistema IoT Control Plane?",
        "Quali sono i requisiti hardware minimi?",
        "Come si gestiscono gli errori di connessione?",
        "Che cos'è l'autenticazione nel sistema?",
        "Come si monitora lo stato del sistema?"
    ]
    
    print("\n" + "="*60)
    print("🤖 SISTEMA RAG PRONTO - Esempi di Query")
    print("="*60)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 50)
        
        try:
            # Esegui la query
            response = rag_pipeline.query(
                query=query,
                top_k=3,
                include_metadata=True
            )
            
            # Mostra la risposta
            print(f"🎯 Risposta (Confidence: {response.confidence:.2f}):")
            print(response.answer)
            
            print(f"\n📚 Fonti utilizzate ({len(response.sources)}):")
            for source in response.sources[:3]:  # Mostra prime 3 fonti
                print(f"  - {source}")
            
            print(f"\n⏱️  Tempo di elaborazione: {response.processing_time:.2f}s")
            
            # Mostra statistiche dettagliate per la prima query
            if i == 1 and response.metadata.get("retrieval_stats"):
                print(f"\n📊 Statistiche dettagliate:")
                stats = response.metadata["retrieval_stats"]
                print(f"  - Score fusion medio: {stats.get('avg_fusion_score', 0):.3f}")
                print(f"  - Score vettoriale medio: {stats.get('avg_vector_score', 0):.3f}")
                print(f"  - Score BM25 medio: {stats.get('avg_bm25_score', 0):.3f}")
            
        except Exception as e:
            logger.error(f"Errore nella query {i}: {e}")
            print(f"❌ Errore: {e}")
    
    # 7. Genera un riassunto dei documenti
    print(f"\n{'='*60}")
    print("📄 RIASSUNTO DOCUMENTI")
    print("="*60)
    
    try:
        summary = rag_pipeline.get_document_summary(summary_type="comprehensive")
        print(summary.answer)
    except Exception as e:
        logger.error(f"Errore nel riassunto: {e}")
        print(f"❌ Errore nel generare riassunto: {e}")
    
    # 8. Query interattiva
    print(f"\n{'='*60}")
    print("💬 MODALITÀ INTERATTIVA")
    print("="*60)
    print("Scrivi le tue domande (digita 'quit' per uscire):")
    
    while True:
        try:
            user_query = input("\n🔍 Domanda: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_query:
                continue
            
            print("🤔 Elaborando...")
            response = rag_pipeline.query(user_query, top_k=3)
            
            print(f"\n🎯 Risposta:")
            print(response.answer)
            
            if response.sources:
                print(f"\n📚 Fonti: {', '.join(response.sources[:2])}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Errore nella query interattiva: {e}")
            print(f"❌ Errore: {e}")
    
    print("\n👋 Grazie per aver utilizzato il sistema RAG!")
    logger.info("=== FINE SESSIONE RAG ===")


def demo_batch_processing():
    """Dimostra il processing in batch di multiple query"""
    
    logger = setup_logging(log_level="INFO", console_output=True)
    config = get_default_config()
    
    # Valida API keys
    if not all(validate_api_keys(["OPENAI_API_KEY", "GOOGLE_API_KEY"]).values()):
        print("❌ API keys mancanti")
        return
    
    # Inizializza pipeline
    rag_pipeline = create_rag_pipeline(config)
    
    # Processa documenti (usa cache se disponibile)
    rag_pipeline.process_documents(["data"], force_reprocess=False)
    
    # Query in batch
    batch_queries = [
        "Cos'è il sistema IoT Control Plane?",
        "Come si installa il software?",
        "Quali sono le funzionalità principali?",
        "Come si risolve un errore di connessione?",
        "Dove si trovano i log del sistema?"
    ]
    
    print("🔄 Processing batch di query...")
    responses = rag_pipeline.batch_query(batch_queries, top_k=2)
    
    print(f"\n📊 Risultati batch ({len(responses)} query processate):")
    for i, (query, response) in enumerate(zip(batch_queries, responses), 1):
        print(f"\n{i}. {query}")
        print(f"   Risposta: {response.answer[:100]}...")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Tempo: {response.processing_time:.2f}s")


if __name__ == "__main__":
    # Controlla se esiste il file .env
    if not Path(".env").exists():
        print("⚠️  File .env non trovato!")
        print("Crea un file .env con le seguenti variabili:")
        print("OPENAI_API_KEY=your_openai_api_key")
        print("GOOGLE_API_KEY=your_google_api_key")
        print("\nOppure imposta le variabili d'ambiente direttamente.")
        
        # Chiedi se continuare comunque
        choice = input("\nContinuare comunque? (y/n): ").lower()
        if choice != 'y':
            exit(1)
    
    # Esegui l'esempio principale
    main()
    
    # Opzionalmente esegui il demo batch
    print("\n" + "="*60)
    choice = input("Vuoi eseguire il demo batch processing? (y/n): ").lower()
    if choice == 'y':
        demo_batch_processing()
