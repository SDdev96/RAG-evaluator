# 🚀 Guida di Installazione - Sistema RAG Avanzato

## 📋 Prerequisiti

### Requisiti di Sistema
- **Python**: 3.8 o superiore
- **RAM**: Minimo 8GB (consigliato 16GB)
- **Spazio Disco**: Almeno 2GB liberi
- **Connessione Internet**: Per scaricare modelli e API calls

### Account API Richiesti
1. **OpenAI Account**: Per embeddings e semantic chunking
   - Vai su [platform.openai.com](https://platform.openai.com)
   - Crea account e ottieni API key
   
2. **Google Cloud Account**: Per Gemini API
   - Vai su [console.cloud.google.com](https://console.cloud.google.com)
   - Abilita Gemini API e ottieni API key

## 🔧 Installazione Passo-Passo

### 1. Clona il Repository
```bash
git clone <repository-url>
cd RAG-evaluator
```

### 2. Crea Ambiente Virtuale (Consigliato)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Installa Dipendenze Base
```bash
pip install -r requirements.txt
```

### 4. Installa Dipendenze Opzionali

#### Per Document Processing Avanzato (Docling)
```bash
pip install docling
```

#### Per Funzionalità Avanzate
```bash
pip install nltk spacy transformers
```

#### Per Sviluppo
```bash
pip install pytest black flake8 loguru
```

### 5. Configura API Keys

#### Opzione A: File .env (Consigliato)
```bash
# Copia il template
cp .env.example .env

# Modifica il file .env con le tue API keys
OPENAI_API_KEY=sk-your-openai-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

#### Opzione B: Variabili d'Ambiente di Sistema
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-your-openai-key-here"
$env:GOOGLE_API_KEY="your-google-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="sk-your-openai-key-here"
export GOOGLE_API_KEY="your-google-api-key-here"
```

## ✅ Verifica Installazione

### Test Rapido
```bash
python quick_test.py
```

Questo script verifica:
- ✅ Import di tutti i moduli
- ✅ Configurazione API keys
- ✅ Funzionamento componenti base
- ✅ Processing documenti di test

### Test Completo
```bash
python main.py --docs data/ --query "Test di funzionamento"
```

## 📁 Struttura Directory

Dopo l'installazione, la struttura dovrebbe essere:
```
RAG-evaluator/
├── config/                    # Configurazioni
├── src/                       # Codice sorgente
│   ├── document_processing/
│   ├── chunking/
│   ├── query_handling/
│   ├── retrieval/
│   ├── generation/
│   ├── pipeline/
│   └── utils/
├── data/                      # Documenti input
├── processed/                 # Documenti processati (auto-creata)
├── vector_stores/            # Indici FAISS (auto-creata)
├── cache/                    # Cache pipeline (auto-creata)
├── logs/                     # File di log (auto-creata)
├── .env                      # API keys (da creare)
├── requirements.txt
├── main.py
├── example_usage.py
├── quick_test.py
└── README.md
```

## 🚨 Risoluzione Problemi

### Errore: "ModuleNotFoundError"
```bash
# Verifica ambiente virtuale attivo
pip list

# Reinstalla dipendenze
pip install -r requirements.txt --force-reinstall
```

### Errore: "API Key not found"
```bash
# Verifica file .env
cat .env  # Linux/Mac
type .env  # Windows

# Verifica variabili d'ambiente
echo $OPENAI_API_KEY  # Linux/Mac
echo $env:OPENAI_API_KEY  # Windows PowerShell
```

### Errore: "Docling not found"
```bash
# Installa Docling (opzionale)
pip install docling

# Il sistema funziona anche senza Docling usando PyPDF2
```

### Errore: "FAISS installation failed"
```bash
# Su Windows, prova:
pip install faiss-cpu --no-cache-dir

# Su Linux/Mac:
conda install faiss-cpu -c conda-forge
```

### Errore: "Out of memory"
```bash
# Riduci parametri in config/config.py:
# - chunk_size = 500 (invece di 1000)
# - num_hypothetical_questions = 2 (invece di 3)
# - top_k = 3 (invece di 5)
```

## 🔧 Configurazione Avanzata

### Personalizza Configurazione
Modifica `config/config.py` per:
- Cambiare modelli utilizzati
- Modificare parametri di chunking
- Regolare pesi fusion retrieval
- Personalizzare generazione

### Esempio Configurazione Custom
```python
from config.config import RAGConfig, ChunkingConfig

config = RAGConfig(
    chunking=ChunkingConfig(
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=2,
        min_chunk_size=200,
        max_chunk_size=1500
    )
)
```

## 📊 Monitoraggio Performance

### Abilita Logging Dettagliato
```python
from src.utils.logging_config import setup_logging

logger = setup_logging(
    log_level="DEBUG",
    log_file="detailed.log"
)
```

### Metriche Disponibili
- Tempo processing per documento
- Numero chunks generati
- Score retrieval (vettoriale, BM25, fusion)
- Confidence risposte
- Utilizzo memoria

## 🔄 Aggiornamenti

### Aggiorna Dipendenze
```bash
pip install -r requirements.txt --upgrade
```

### Pulisci Cache
```bash
python -c "from src.pipeline.advanced_rag_pipeline import create_rag_pipeline; create_rag_pipeline().clear_cache()"
```

## 🆘 Supporto

### Log di Debug
I log dettagliati sono salvati in `logs/`:
- `rag_system_YYYYMMDD_HHMMSS.log`: Log completi
- Console output per errori immediati

### Informazioni Sistema
```bash
python -c "
import sys
print(f'Python: {sys.version}')
print(f'Platform: {sys.platform}')

import pkg_resources
for pkg in ['langchain', 'openai', 'faiss-cpu', 'google-generativeai']:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f'{pkg}: {version}')
    except:
        print(f'{pkg}: Not installed')
"
```

### Test Componenti Singoli
```bash
# Test solo document processing
python -c "from src.document_processing.docling_processor import create_processor; print('✅ Document processing OK')"

# Test solo semantic chunking
python -c "from src.chunking.semantic_chunker import create_semantic_chunker; print('✅ Chunking OK')"

# Test solo HyPE
python -c "from src.query_handling.hype_processor import create_hype_processor; print('✅ HyPE OK')"
```

## 🎯 Prossimi Passi

Dopo l'installazione riuscita:

1. **Testa con i tuoi documenti**:
   ```bash
   python main.py --docs /path/to/your/documents --interactive
   ```

2. **Esplora esempi**:
   ```bash
   python example_usage.py
   ```

3. **Personalizza configurazione** in `config/config.py`

4. **Leggi documentazione completa** in `README.md`

---

🎉 **Installazione completata!** Il sistema RAG è pronto per l'uso.
