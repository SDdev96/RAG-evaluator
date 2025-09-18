#!/usr/bin/env python3
"""
Test rapido del sistema RAG per verificare che tutti i componenti funzionino
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()

def test_imports():
    """Test degli import di tutti i moduli"""
    print("🔍 Testing imports...")
    
    try:
        from config.config import get_default_config
        print("✅ Config module")
        
        from src.document_processing import create_processor
        print("✅ Document processing module (PyPDF2)")
        
        from src.chunking.semantic_chunker import create_semantic_chunker
        print("✅ Semantic chunking module")
        
        from src.query_handling.query_transformations import create_query_transformer
        print("✅ Query Transformations module")
        
        from src.retrieval.fusion_retriever import create_fusion_retriever
        print("✅ Fusion retrieval module")
        
        from src.generation.gemini_generator import create_gemini_generator
        print("✅ Gemini generation module")
        
        from src.pipeline.advanced_rag_pipeline import create_rag_pipeline
        print("✅ Pipeline module")
        
        from src.utils.logging_config import setup_logging
        from src.utils.helpers import validate_api_keys
        print("✅ Utils modules")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_api_keys():
    """Test delle API keys"""
    print("\n🔑 Testing API keys...")
    
    from src.utils.helpers import validate_api_keys
    
    required_keys = ["GOOGLE_API_KEY"]
    validation = validate_api_keys(required_keys)
    
    all_valid = True
    for key, is_valid in validation.items():
        if is_valid:
            print(f"✅ {key} configurata")
        else:
            print(f"❌ {key} mancante")
            all_valid = False
    
    return all_valid


def test_configuration():
    """Test della configurazione"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from config.config import get_default_config
        
        config = get_default_config()
        print(f"✅ Configurazione caricata")
        print(f"  - Document processing: {config.document_processing.input_dir}")
        print(f"  - Chunking method: {config.chunking.breakpoint_threshold_type}")
        print(f"  - Query transformations: decompose={config.query_transformations.enable_decompose}, rewrite={config.query_transformations.enable_rewrite}, expand={config.query_transformations.enable_expand}")
        print(f"  - Fusion weights: {config.fusion_retrieval.vector_weight}/{config.fusion_retrieval.bm25_weight}")
        print(f"  - Generation model: {config.generation.model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_document_processing():
    """Test del document processing"""
    print("\n📄 Testing document processing...")
    
    try:
        from src.document_processing import create_processor
        from config.config import DocumentProcessingConfig
        
        # Controlla se esiste il PDF di test
        pdf_path = Path("data/X.AMN.PNRR.U.001.07.00 Manuale utente IoT Control Plane.pdf")
        
        if not pdf_path.exists():
            print(f"⚠️ PDF di test non trovato: {pdf_path}")
            print("   Creando un documento di test...")
            
            # Crea un documento di test semplice
            test_doc_path = Path("data/test_document.txt")
            test_doc_path.parent.mkdir(exist_ok=True)
            
            with open(test_doc_path, 'w', encoding='utf-8') as f:
                f.write("""# Documento di Test per Sistema RAG

## Introduzione
Questo è un documento di test per verificare il funzionamento del sistema RAG avanzato.

## Configurazione Sistema
Per configurare il sistema IoT Control Plane, seguire questi passaggi:
1. Installare il software base
2. Configurare le credenziali di accesso
3. Verificare la connessione di rete

## Risoluzione Problemi
In caso di errori di connessione:
- Verificare le impostazioni di rete
- Controllare le credenziali
- Riavviare il servizio se necessario

## Monitoraggio
Il sistema fornisce strumenti di monitoraggio per:
- Stato delle connessioni
- Performance del sistema
- Log degli eventi
""")
            
            pdf_path = test_doc_path
        
        # Test del processor
        config = DocumentProcessingConfig()
        processor = create_processor(config)
        
        processed_doc = processor.process_document(str(pdf_path))
        
        print(f"✅ Documento processato: {processed_doc.metadata.get('title', 'Unknown')}")
        print(f"  - Contenuto: {len(processed_doc.content)} caratteri")
        print(f"  - Metodo: {processed_doc.metadata.get('processing_method', 'Unknown')}")
        
        return True, processed_doc
        
    except Exception as e:
        print(f"❌ Document processing error: {e}")
        return False, None


def test_semantic_chunking(processed_doc):
    """Test del semantic chunking"""
    print("\n🔪 Testing semantic chunking...")
    
    try:
        from src.chunking.semantic_chunker import create_semantic_chunker
        from config.config import ChunkingConfig, get_default_config
        from src.embeddings.provider import EmbeddingsProvider
        
        if not processed_doc:
            print("❌ Nessun documento processato disponibile")
            return False, []
        
        config = ChunkingConfig()
        # Inizializza provider centralizzato dagli defaults
        rag_cfg = get_default_config()
        provider = EmbeddingsProvider.get(rag_cfg.embeddings)
        chunker = create_semantic_chunker(config, embeddings_provider=provider)
        
        # Usa un testo più breve per il test se necessario
        test_content = processed_doc.content[:2000] if len(processed_doc.content) > 2000 else processed_doc.content
        
        chunks = chunker.chunk_text(test_content, {"source": "test"})
        
        print(f"✅ Creati {len(chunks)} chunks semantici")
        if chunks:
            print(f"  - Chunk medio: {sum(len(c.content) for c in chunks) // len(chunks)} caratteri")
            print(f"  - Metodo: {chunks[0].metadata.get('chunking_method', 'Unknown')}")
        
        return True, chunks
        
    except Exception as e:
        print(f"❌ Semantic chunking error: {e}")
        return False, []


def test_basic_pipeline():
    """Test base della pipeline senza API calls costose"""
    print("\n🔧 Testing basic pipeline initialization...")
    
    try:
        from src.pipeline.advanced_rag_pipeline import create_rag_pipeline
        from config.config import get_default_config
        
        config = get_default_config()
        
        # Modifica config per test più veloce
        config.query_transformations.max_transformations = 3  # Riduci per test
        
        pipeline = create_rag_pipeline(config)
        
        print("✅ Pipeline inizializzata correttamente")
        print(f"  - Document processor: {type(pipeline.doc_processor).__name__}")
        print(f"  - Chunker: {type(pipeline.chunker).__name__}")
        print(f"  - Query transformer: {type(pipeline.query_transformer).__name__}")
        print(f"  - Retriever: {type(pipeline.retriever).__name__}")
        print(f"  - Generator: {type(pipeline.generator).__name__}")
        
        return True, pipeline
        
    except Exception as e:
        print(f"❌ Pipeline initialization error: {e}")
        return False, None


def main():
    """Test principale"""
    print("🚀 QUICK TEST - Sistema RAG Avanzato")
    print("=" * 50)
    
    # Test degli import
    if not test_imports():
        print("\n❌ Test fallito: problemi con gli import")
        sys.exit(1)
    
    # Test API keys
    api_keys_ok = test_api_keys()
    if not api_keys_ok:
        print("\n⚠️ API keys mancanti - alcuni test potrebbero fallire")
    
    # Test configurazione
    if not test_configuration():
        print("\n❌ Test fallito: problemi con la configurazione")
        sys.exit(1)
    
    # Test document processing
    doc_ok, processed_doc = test_document_processing()
    if not doc_ok:
        print("\n❌ Test fallito: problemi con document processing")
        sys.exit(1)
    
    # Test semantic chunking
    chunk_ok, chunks = test_semantic_chunking(processed_doc)
    if not chunk_ok:
        print("\n⚠️ Warning: problemi con semantic chunking")
    
    # Test basic pipeline
    pipeline_ok, pipeline = test_basic_pipeline()
    if not pipeline_ok:
        print("\n❌ Test fallito: problemi con la pipeline")
        sys.exit(1)
    
    # Riepilogo
    print("\n" + "=" * 50)
    print("📊 RIEPILOGO TEST")
    print("=" * 50)
    print("✅ Import modules: OK")
    print(f"{'✅' if api_keys_ok else '⚠️'} API Keys: {'OK' if api_keys_ok else 'Parziale'}")
    print("✅ Configuration: OK")
    print("✅ Document Processing: OK")
    print("✅ Semantic Chunking: OK" if chunk_ok else "⚠️ Semantic Chunking: Issues")
    print("✅ Pipeline Initialization: OK")
    
    if api_keys_ok:
        print("\n🎉 Tutti i test superati! Il sistema è pronto per l'uso.")
        print("\nPer testare completamente il sistema:")
        print("  python main.py --docs data/ --interactive")
    else:
        print("\n⚠️ Test base superati, ma configura le API keys per funzionalità complete.")
        print("\nConfigura nel file .env:")
        print("  GOOGLE_API_KEY=your_key_here")
        print("  # Opzionale per Hugging Face (modelli privati o limiti più alti):")
        print("  HUGGINGFACEHUB_API_TOKEN=your_hf_token_here")
    
    print("\n📚 Per maggiori informazioni consulta il README.md")


if __name__ == "__main__":
    main()
