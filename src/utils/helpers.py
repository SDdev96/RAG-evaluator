"""
Funzioni di utilità per il sistema RAG
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime


def load_environment_variables(env_file: str = ".env") -> Dict[str, str]:
    """
    Carica le variabili d'ambiente da un file .env
    
    Args:
        env_file: Percorso al file .env
        
    Returns:
        Dict[str, str]: Dizionario delle variabili caricate
    """
    env_vars = {}
    env_path = Path(env_file)
    
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    env_vars[key] = value
                    os.environ[key] = value
    
    return env_vars


def validate_api_keys(required_keys: List[str]) -> Dict[str, bool]:
    """
    Valida che le API keys richieste siano presenti
    
    Args:
        required_keys: Lista delle chiavi API richieste
        
    Returns:
        Dict[str, bool]: Stato di validazione per ogni chiave
    """
    validation_results = {}
    
    for key in required_keys:
        value = os.getenv(key)
        validation_results[key] = bool(value and len(value.strip()) > 0)
    
    return validation_results


def create_file_hash(file_path: Union[str, Path]) -> str:
    """
    Crea un hash MD5 di un file
    
    Args:
        file_path: Percorso al file
        
    Returns:
        str: Hash MD5 del file
    """
    hash_md5 = hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()


def save_json(data: Dict[str, Any], file_path: Union[str, Path], 
              indent: int = 2, ensure_ascii: bool = False) -> bool:
    """
    Salva dati in formato JSON
    
    Args:
        data: Dati da salvare
        file_path: Percorso del file
        indent: Indentazione JSON
        ensure_ascii: Se forzare ASCII
        
    Returns:
        bool: True se salvato con successo
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
        
        return True
    except Exception:
        return False


def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Carica dati da un file JSON
    
    Args:
        file_path: Percorso del file
        
    Returns:
        Optional[Dict[str, Any]]: Dati caricati o None se errore
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def format_file_size(size_bytes: int) -> str:
    """
    Formatta la dimensione di un file in modo leggibile
    
    Args:
        size_bytes: Dimensione in bytes
        
    Returns:
        str: Dimensione formattata
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def clean_text(text: str) -> str:
    """
    Pulisce un testo rimuovendo caratteri indesiderati
    
    Args:
        text: Testo da pulire
        
    Returns:
        str: Testo pulito
    """
    if not text:
        return ""
    
    # Rimuovi caratteri di controllo
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    # Normalizza spazi bianchi
    text = ' '.join(text.split())
    
    # Rimuovi spazi multipli
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Tronca un testo alla lunghezza massima specificata
    
    Args:
        text: Testo da troncare
        max_length: Lunghezza massima
        suffix: Suffisso da aggiungere se troncato
        
    Returns:
        str: Testo troncato
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Ottiene informazioni dettagliate su un file
    
    Args:
        file_path: Percorso del file
        
    Returns:
        Dict[str, Any]: Informazioni del file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"exists": False}
    
    stat = file_path.stat()
    
    return {
        "exists": True,
        "name": file_path.name,
        "size": stat.st_size,
        "size_formatted": format_file_size(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "created": datetime.fromtimestamp(stat.st_ctime),
        "extension": file_path.suffix,
        "is_file": file_path.is_file(),
        "is_dir": file_path.is_dir(),
        "hash": create_file_hash(file_path) if file_path.is_file() else None
    }


def create_directory_structure(base_path: Union[str, Path], 
                             structure: Dict[str, Any]) -> bool:
    """
    Crea una struttura di directory basata su un dizionario
    
    Args:
        base_path: Percorso base
        structure: Struttura delle directory
        
    Returns:
        bool: True se creato con successo
    """
    try:
        base_path = Path(base_path)
        
        for name, content in structure.items():
            current_path = base_path / name
            
            if isinstance(content, dict):
                # È una directory
                current_path.mkdir(parents=True, exist_ok=True)
                create_directory_structure(current_path, content)
            else:
                # È un file
                current_path.parent.mkdir(parents=True, exist_ok=True)
                if content is not None:
                    with open(current_path, 'w', encoding='utf-8') as f:
                        f.write(str(content))
        
        return True
    except Exception:
        return False


def measure_execution_time(func):
    """
    Decorator per misurare il tempo di esecuzione di una funzione
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{func.__name__} eseguita in {execution_time:.4f} secondi")
        
        return result
    
    return wrapper


def batch_process(items: List[Any], batch_size: int = 10):
    """
    Generator per processare elementi in batch
    
    Args:
        items: Lista di elementi da processare
        batch_size: Dimensione del batch
        
    Yields:
        List[Any]: Batch di elementi
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Divisione sicura che gestisce la divisione per zero
    
    Args:
        a: Numeratore
        b: Denominatore
        default: Valore di default se divisione per zero
        
    Returns:
        float: Risultato della divisione o valore di default
    """
    try:
        return a / b if b != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalizza un punteggio tra min_val e max_val
    
    Args:
        score: Punteggio da normalizzare
        min_val: Valore minimo
        max_val: Valore massimo
        
    Returns:
        float: Punteggio normalizzato
    """
    return max(min_val, min(max_val, score))
