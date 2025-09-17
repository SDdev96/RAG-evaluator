# 🚀 Sistema RAG Avanzato

Un sistema di Retrieval-Augmented Generation (RAG) modulare che integra tecniche avanzate per il processing, chunking, retrieval e generazione di risposte da documenti tecnici.

## 🎯 Caratteristiche Principali

### 🔄 Pipeline Completa
- **Document Processing**: Conversione documenti con Docling (PDF → Markdown strutturato)
- **Semantic Chunking**: Divisione intelligente del testo basata su similarità semantica
- **HyPE**: Hypothetical Prompt Embeddings per query handling ottimizzato
- **Fusion Retrieval**: Combinazione di ricerca vettoriale (FAISS) e keyword-based (BM25)
- **Gemini Generation**: Generazione risposte con Google Gemini API

### 🏗️ Architettura Modulare
```
src/
├── document_processing/    # Docling processor
├── chunking/              # Semantic chunking
├── query_handling/        # HyPE processor
├── retrieval/             # Fusion retrieval
├── generation/            # Gemini generator
├── pipeline/              # Pipeline principale
└── utils/                 # Utilità e helpers
```

## 🛠️ Installazione

### 1. Clona il repository
```bash
git clone <repository-url>
cd RAG-evaluator
```

### 2. Installa le dipendenze
```bash
pip install -r requirements.txt
```

### 3. Configura le API Keys
Crea un file `.env` nella root del progetto:
```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 4. (Opzionale) Installa Docling
Per il processing avanzato dei PDF:
```bash
pip install docling
```

## 🚀 Utilizzo Rapido

### Esempio Base
```python
from config.config import get_default_config
from src.pipeline.advanced_rag_pipeline import create_rag_pipeline

# Inizializza la pipeline
config = get_default_config()
rag_pipeline = create_rag_pipeline(config)

# Processa i documenti
processed_docs = rag_pipeline.process_documents(["data/"])

# Esegui una query
response = rag_pipeline.query("Come si configura il sistema?")
print(response.answer)
```

### Esecuzione dell'Esempio Completo
```bash
python example_usage.py
```

## 📋 Flusso di Esecuzione

### 0. Document Processing
- **Input**: PDF, DOCX, TXT, MD
- **Processo**: Docling converte documenti in Markdown strutturato
- **Output**: Documenti processati con metadati

### 1. Semantic Chunking
- **Input**: Testo strutturato
- **Processo**: LangChain SemanticChunker divide il testo in chunks semanticamente coerenti
- **Parametri**: `percentile`, `standard_deviation`, `interquartile`
- **Output**: Chunks con boundaries semantiche naturali

### 2. Query Handling (HyPE)
- **Input**: Chunks semantici
- **Processo**: Genera domande ipotetiche per ogni chunk usando LLM
- **Benefici**: Migliora l'allineamento query-documento
- **Output**: Chunks arricchiti con embeddings multipli

### 3. Fusion Retrieval
- **Input**: Query utente + indici costruiti
- **Processo**: Combina ricerca vettoriale (FAISS) e keyword (BM25)
- **Formula**: `score = α × vector_score + β × bm25_score`
- **Output**: Risultati ranked con score fusion

### 4. Generation
- **Input**: Query + chunks recuperati
- **Processo**: Google Gemini genera risposta basata sul contesto
- **Output**: Risposta strutturata con fonti e confidence

## ⚙️ Configurazione

### Configurazione Base
```python
from config.config import RAGConfig, DocumentProcessingConfig, ChunkingConfig

config = RAGConfig(
    document_processing=DocumentProcessingConfig(
        input_dir="data",
        output_dir="processed"
    ),
    chunking=ChunkingConfig(
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    ),
    # ... altre configurazioni
)
```

### Parametri Principali

#### Semantic Chunking
- `breakpoint_threshold_type`: `"percentile"`, `"standard_deviation"`, `"interquartile"`
- `breakpoint_threshold_amount`: Soglia per la divisione (es. 90 per percentile)
- `min_chunk_size`: Dimensione minima chunk (default: 100)
- `max_chunk_size`: Dimensione massima chunk (default: 2000)

#### HyPE
- `num_hypothetical_questions`: Numero domande per chunk (default: 3)
- `language_model`: Modello per generazione domande (default: "gpt-3.5-turbo")
- `temperature`: Creatività generazione (default: 0.7)

#### Fusion Retrieval
- `vector_weight`: Peso ricerca vettoriale (default: 0.7)
- `bm25_weight`: Peso ricerca keyword (default: 0.3)
- `top_k`: Numero risultati da restituire (default: 5)

#### Generation
- `model_name`: Modello Gemini (default: "gemini-1.5-flash")
- `temperature`: Creatività risposta (default: 0.3)
- `max_tokens`: Lunghezza massima risposta (default: 1000)

## 📊 Monitoraggio e Logging

### Logging Configurabile
```python
from src.utils.logging_config import setup_logging

logger = setup_logging(
    log_level="INFO",
    log_file="rag_system.log",
    console_output=True
)
```

### Statistiche Disponibili
- Tempo di processing per fase
- Numero chunks generati
- Score di retrieval (vettoriale, BM25, fusion)
- Confidence delle risposte
- Utilizzo cache

## 🔧 Funzionalità Avanzate

### Cache Intelligente
- Salvataggio automatico stato pipeline
- Ricaricamento rapido da cache
- Invalidazione basata su modifiche documenti

### Processing Parallelo
- HyPE processing con ThreadPoolExecutor
- Batch processing per multiple query
- Ottimizzazione performance

### Gestione Errori
- Fallback automatici per ogni componente
- Logging dettagliato degli errori
- Graceful degradation

## 📁 Struttura File

```
RAG-evaluator/
├── config/
│   └── config.py              # Configurazioni sistema
├── src/
│   ├── document_processing/
│   │   └── docling_processor.py
│   ├── chunking/
│   │   └── semantic_chunker.py
│   ├── query_handling/
│   │   └── hype_processor.py
│   ├── retrieval/
│   │   └── fusion_retriever.py
│   ├── generation/
│   │   └── gemini_generator.py
│   ├── pipeline/
│   │   └── advanced_rag_pipeline.py
│   └── utils/
│       ├── logging_config.py
│       └── helpers.py
├── data/                      # Documenti input
├── processed/                 # Documenti processati
├── vector_stores/            # Indici FAISS
├── cache/                    # Cache pipeline
├── logs/                     # File di log
├── requirements.txt
├── example_usage.py
└── README.md
```

## 🧪 Testing

### Test Rapido
```python
# Test processing singolo documento
from src.document_processing.docling_processor import create_processor

processor = create_processor()
doc = processor.process_document("data/manual.pdf")
print(f"Processato: {doc.metadata['title']}")
```

### Test Pipeline Completa
```bash
python example_usage.py
```

## 🔍 Esempi di Query

Il sistema è ottimizzato per documenti tecnici e supporta query come:

- **Configurazione**: "Come si configura il sistema IoT Control Plane?"
- **Troubleshooting**: "Come si risolve un errore di connessione?"
- **Funzionalità**: "Quali sono le funzionalità di monitoraggio?"
- **Requisiti**: "Quali sono i requisiti hardware minimi?"
- **Procedure**: "Come si esegue il backup dei dati?"

## 📈 Performance

### Metriche Tipiche
- **Processing iniziale**: 30-60 secondi per documento medio
- **Query response**: 2-5 secondi
- **Cache hit**: <1 secondo
- **Accuracy**: Dipende dalla qualità dei documenti e tuning parametri

### Ottimizzazioni
- Cache persistente per evitare riprocessing
- Indici FAISS ottimizzati per similarità coseno
- Processing parallelo per HyPE
- Batch processing per multiple query

## 🛡️ Sicurezza

- API keys gestite tramite variabili d'ambiente
- Validazione input per prevenire injection
- Logging sicuro (no API keys nei log)
- Gestione errori senza esposizione dettagli interni

## 🤝 Contribuire

1. Fork del repository
2. Crea branch per feature (`git checkout -b feature/AmazingFeature`)
3. Commit modifiche (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri Pull Request

## 📝 Licenza

Questo progetto è distribuito sotto licenza MIT. Vedi `LICENSE` per dettagli.

## 🙏 Riconoscimenti

- **Docling**: Document processing avanzato
- **LangChain**: Framework per semantic chunking
- **FAISS**: Ricerca vettoriale efficiente
- **Google Gemini**: Generazione risposte di alta qualità
- **OpenAI**: Embeddings per semantic understanding

## 📞 Supporto

Per domande, problemi o suggerimenti:
1. Apri un issue su GitHub
2. Consulta i log in `logs/` per debugging
3. Verifica la configurazione in `config/config.py`

---

**Nota**: Questo sistema è progettato per documenti tecnici in italiano. Per altri domini o lingue, potrebbe essere necessario un tuning dei parametri.
