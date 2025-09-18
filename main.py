#!/usr/bin/env python3
"""
Main entry point per il sistema RAG avanzato
"""
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

from config.config import get_default_config
from src.pipeline.advanced_rag_pipeline import create_rag_pipeline
from src.utils.logging_config import setup_logging
from src.utils.helpers import validate_api_keys


def setup_environment():
    """Setup dell'ambiente e validazione"""
    # Setup logging
    logger = setup_logging(
        log_level="INFO",
        console_output=True
    )
    
    # Valida API keys (per default servono solo quelle di Google Gemini)
    required_keys = ["GOOGLE_API_KEY"]
    api_validation = validate_api_keys(required_keys)
    
    missing_keys = [key for key, valid in api_validation.items() if not valid]
    
    if missing_keys:
        logger.error(f"API keys mancanti: {', '.join(missing_keys)}")
        print("\n❌ Configurazione incompleta!")
        print("Configura le seguenti variabili d'ambiente:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nPuoi:")
        print("1. Creare un file .env (vedi .env.example)")
        print("2. Impostare le variabili d'ambiente del sistema")
        return False, logger
    
    logger.info("✅ Ambiente configurato correttamente")
    return True, logger


def process_documents(pipeline, document_paths, force_reprocess=False):
    """Processa i documenti"""
    try:
        processed_docs = pipeline.process_documents(
            document_paths,
            force_reprocess=force_reprocess
        )
        print(f"✅ Processati {len(processed_docs)} documenti")
        return True
    except Exception as e:
        print(f"❌ Errore nel processing: {e}")
        return False


def interactive_mode(pipeline):
    """Modalità interattiva per query"""
    print("\n" + "="*60)
    print("💬 MODALITÀ INTERATTIVA")
    print("="*60)
    print("Comandi disponibili:")
    print("  - Scrivi una domanda per ottenere una risposta")
    print("  - 'summary' per un riassunto dei documenti")
    print("  - 'stats' per statistiche del sistema")
    print("  - 'help' per questo messaggio")
    print("  - 'quit' per uscire")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n🔍 > ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'help':
                print("Comandi: query normale, 'summary', 'stats', 'quit'")
                continue
            elif user_input.lower() == 'summary':
                print("📄 Generando riassunto...")
                try:
                    summary = pipeline.get_document_summary("comprehensive")
                    print(f"\n📋 Riassunto:\n{summary.answer}")
                except Exception as e:
                    print(f"❌ Errore nel riassunto: {e}")
                continue
            elif user_input.lower() == 'stats':
                if pipeline.state:
                    print(f"\n📊 Statistiche:")
                    print(f"  - Documenti processati: {len(pipeline.state.processed_documents)}")
                    print(f"  - Chunks semantici: {len(pipeline.state.semantic_chunks)}")
                    print(f"  - Chunks arricchiti: {len(pipeline.state.enriched_chunks)}")
                    total_questions = sum(len(chunk.hypothetical_questions) for chunk in pipeline.state.enriched_chunks)
                    print(f"  - Domande ipotetiche: {total_questions}")
                else:
                    print("❌ Nessun dato processato")
                continue
            elif not user_input:
                continue
            
            # Esegui query normale
            print("🤔 Elaborando...")
            response = pipeline.query(user_input, top_k=8)
            
            print(f"\n🎯 Risposta (Confidence: {response.confidence:.2f}):")
            print(response.answer)
            
            if response.sources:
                print(f"\n📚 Fonti principali:")
                for i, source in enumerate(response.sources[:8], 1):
                    print(f"  {i}. {source}")
            
            print(f"\n⏱️ Tempo: {response.processing_time:.2f}s")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Errore: {e}")
    
    print("\n👋 Arrivederci!")


def batch_mode(pipeline, queries_file):
    """Modalità batch da file"""
    try:
        queries_path = Path(queries_file)
        if not queries_path.exists():
            print(f"❌ File non trovato: {queries_file}")
            return
        
        with open(queries_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        if not queries:
            print("❌ Nessuna query trovata nel file")
            return
        
        print(f"🔄 Processando {len(queries)} query da {queries_file}")
        
        responses = pipeline.batch_query(queries, top_k=8)
        
        # Salva risultati
        output_file = queries_path.with_suffix('.results.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (query, response) in enumerate(zip(queries, responses), 1):
                f.write(f"Query {i}: {query}\n")
                f.write(f"Risposta: {response.answer}\n")
                f.write(f"Confidence: {response.confidence:.2f}\n")
                f.write(f"Fonti: {', '.join(response.sources)}\n")
                f.write(f"Tempo: {response.processing_time:.2f}s\n")
                f.write("-" * 80 + "\n")
        
        print(f"✅ Risultati salvati in: {output_file}")
        
    except Exception as e:
        print(f"❌ Errore nel batch processing: {e}")


def main():
    """Funzione principale"""
    parser = argparse.ArgumentParser(
        description="Sistema RAG Avanzato",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  python main.py --docs data/                    # Processa documenti e modalità interattiva
  python main.py --docs data/ --query "Come..."  # Singola query
  python main.py --batch queries.txt             # Batch da file
  python main.py --interactive                   # Solo modalità interattiva (usa cache)
        """
    )
    
    parser.add_argument(
        '--docs', 
        nargs='+', 
        help='Percorsi ai documenti da processare'
    )
    
    parser.add_argument(
        '--query', 
        help='Singola query da eseguire'
    )
    
    parser.add_argument(
        '--batch', 
        help='File con query per processing batch'
    )
    
    parser.add_argument(
        '--interactive', 
        action='store_true',
        help='Modalità interattiva'
    )
    
    parser.add_argument(
        '--force-reprocess', 
        action='store_true',
        help='Forza il riprocessamento anche se esiste cache'
    )
    
    parser.add_argument(
        '--summary', 
        choices=['brief', 'comprehensive', 'key_points'],
        help='Genera riassunto dei documenti'
    )
    
    args = parser.parse_args()
    
    # Setup ambiente
    env_ok, logger = setup_environment()
    if not env_ok:
        sys.exit(1)
    
    # Inizializza pipeline
    try:
        config = get_default_config()
        pipeline = create_rag_pipeline(config)
        logger.info("Pipeline inizializzata")
    except Exception as e:
        print(f"❌ Errore nell'inizializzazione: {e}")
        sys.exit(1)
    
    # Processa documenti se specificati
    if args.docs:
        print(f"📁 Processando documenti: {args.docs}")
        if not process_documents(pipeline, args.docs, args.force_reprocess):
            sys.exit(1)
    
    # Esegui azioni richieste
    if args.query:
        # Singola query
        print(f"🔍 Query: {args.query}")
        try:
            response = pipeline.query(args.query)
            print(f"\n🎯 Risposta:\n{response.answer}")
            if response.sources:
                print(f"\n📚 Fonti: {', '.join(response.sources)}")
        except Exception as e:
            print(f"❌ Errore nella query: {e}")
    
    elif args.batch:
        # Batch processing
        batch_mode(pipeline, args.batch)
    
    elif args.summary:
        # Genera riassunto
        try:
            print(f"📄 Generando riassunto ({args.summary})...")
            summary = pipeline.get_document_summary(args.summary)
            print(f"\n📋 Riassunto:\n{summary.answer}")
        except Exception as e:
            print(f"❌ Errore nel riassunto: {e}")
    
    elif args.interactive or not any([args.query, args.batch, args.summary]):
        # Modalità interattiva (default se nessun'altra azione)
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()
