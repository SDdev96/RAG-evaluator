"""
Configurazione del logging per il sistema RAG
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    console_output: bool = True
) -> logging.Logger:
    """
    Configura il sistema di logging
    
    Args:
        log_level: Livello di logging (DEBUG, INFO, WARNING, ERROR)
        log_file: Nome del file di log (opzionale)
        log_dir: Directory per i file di log
        console_output: Se mostrare output su console
        
    Returns:
        logging.Logger: Logger configurato
    """
    
    # Crea directory dei log se non esiste
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configura il formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configura il logger root
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Rimuovi handler esistenti
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Handler per console
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(console_handler)
    
    # Handler per file
    if log_file:
        file_path = log_path / log_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = log_path / f"rag_system_{timestamp}.log"
    
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # File sempre con DEBUG
    root_logger.addHandler(file_handler)
    
    # Logger specifico per il sistema RAG
    rag_logger = logging.getLogger('rag_system')
    rag_logger.info(f"Logging configurato - Livello: {log_level}, File: {file_path}")
    
    return rag_logger


def get_logger(name: str) -> logging.Logger:
    """Ottiene un logger con il nome specificato"""
    return logging.getLogger(name)
