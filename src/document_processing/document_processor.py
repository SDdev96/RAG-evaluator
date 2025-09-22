"""
Document Processing usando Docling per convertire PDF in formato strutturato Markdown
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from config.config import DocumentProcessingConfig


@dataclass
class ProcessedDocument:
    """Rappresenta un documento processato"""
    content: str
    metadata: Dict[str, Any]
    source_path: str
    output_path: str


class DocumentProcessor:
    """Processore di documenti usando Docling"""
    
    def __init__(self, config: DocumentProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("Uso di Docling per il processing dei PDF")
        
    def process_document(self, file_path: str) -> ProcessedDocument:
        """
        Processa un singolo documento
        
        Args:
            file_path: Percorso del file da processare
            
        Returns:
            ProcessedDocument: Documento processato
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File non trovato: {file_path}")
        
        # Verifica formato supportato
        if file_path.suffix.lower() not in self.config.supported_formats:
            raise ValueError(f"Formato non supportato: {file_path.suffix}")
        
        # Se esiste già il file processato in Markdown nella cartella di output, evita il re-processing
        try:
            processed_md_path = self._get_output_path(file_path, ".md")
            if processed_md_path.exists():
                self.logger.info(f"Trovato file processato: {processed_md_path}. Salto il processing.")
                print("Trovato file processato: ", processed_md_path)
                with open(processed_md_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                metadata = {
                    "title": file_path.stem,
                    "processing_method": "cached",
                    "format": "markdown"
                }
                return ProcessedDocument(
                    content=content,
                    metadata=metadata,
                    source_path=str(file_path),
                    output_path=str(processed_md_path)
                )
        except Exception as e:
            # In caso di problemi nel leggere la cache, prosegui con il normale processing
            self.logger.warning(f"Impossibile usare il file già processato: {e}. Procedo con il processing.")
        
        self.logger.info(f"Processando documento: {file_path}")
        
        # Usa Docling per PDF e lettura base per testo semplice
        return self._process_with_fallback(file_path)
    
    def _process_with_fallback(self, file_path: Path) -> ProcessedDocument:
        """Processa il documento usando metodi di fallback"""
        if file_path.suffix.lower() == '.pdf':
            return self._process_pdf_with_docling(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            return self._process_text_file(file_path)
        else:
            raise ValueError(f"Formato non supportato per fallback: {file_path.suffix}")

    def _process_pdf_with_docling(self, file_path: Path) -> ProcessedDocument:
        """Processa PDF usando Docling e restituisce contenuto in Markdown.

        Richiede il pacchetto `docling` installato.
        """
        try:
            try:
                from docling.document_converter import DocumentConverter
            except Exception as ie:
                raise ImportError(
                    "Il pacchetto 'docling' non è installato o non è disponibile. "
                    "Aggiungilo ai requisiti ed esegui l'installazione (pip install docling)."
                ) from ie

            converter = DocumentConverter()
            result = converter.convert(str(file_path))

            # Export in Markdown (Docling fornisce exporter dedicato)
            content = result.document.export_to_markdown()

            # Metadati di base; alcune metriche possono non essere sempre disponibili
            pages = None
            try:
                # Docling potrebbe fornire metadati/metrics; fallback a None
                if hasattr(result, "metrics") and isinstance(result.metrics, dict):
                    pages = result.metrics.get("pages")
            except Exception:
                pages = None

            metadata = {
                "title": file_path.stem,
                "pages": pages,
                "processing_method": "docling",
                "format": "markdown"
            }

            # Salva il file processato
            output_path = self._get_output_path(file_path, ".md")
            self._save_processed_content(content, output_path)

            return ProcessedDocument(
                content=content,
                metadata=metadata,
                source_path=str(file_path),
                output_path=str(output_path)
            )

        except Exception as e:
            self.logger.error(f"Errore nel processing PDF con Docling: {e}")
            raise
    
    def _process_text_file(self, file_path: Path) -> ProcessedDocument:
        """Processa file di testo semplici"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            metadata = {
                "title": file_path.stem,
                "processing_method": "text_reader",
                "format": "text"
            }
            
            # Per file .txt, converte in formato markdown base
            if file_path.suffix.lower() == '.txt':
                content = f"# {file_path.stem}\n\n{content}"
                output_path = self._get_output_path(file_path, ".md")
            else:
                output_path = self._get_output_path(file_path, file_path.suffix)
            
            self._save_processed_content(content, output_path)
            
            return ProcessedDocument(
                content=content,
                metadata=metadata,
                source_path=str(file_path),
                output_path=str(output_path)
            )
            
        except Exception as e:
            self.logger.error(f"Errore nel processing file di testo: {e}")
            raise
    
    def _get_output_path(self, input_path: Path, new_suffix: str) -> Path:
        """Genera il percorso di output per il file processato"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = input_path.stem + new_suffix
        return output_dir / output_filename
    
    def _save_processed_content(self, content: str, output_path: Path):
        """Salva il contenuto processato su file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(content)
            self.logger.info(f"Contenuto salvato in: {output_path}")
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio: {e}")
            raise
    
    def process_directory(self, directory_path: str) -> List[ProcessedDocument]:
        """
        Processa tutti i documenti supportati in una directory
        
        Args:
            directory_path: Percorso della directory da processare
            
        Returns:
            List[ProcessedDocument]: Lista dei documenti processati
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory non trovata: {directory}")
        
        processed_docs = []
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.config.supported_formats:
                try:
                    processed_doc = self.process_document(str(file_path))
                    processed_docs.append(processed_doc)
                except Exception as e:
                    self.logger.error(f"Errore nel processare {file_path}: {e}")
                    continue
        self.logger.info(f"Processati {len(processed_docs)} document{'i' if len(processed_docs) > 1 else 'o'} dalla directory {directory}")

        return processed_docs


def create_processor(config: Optional[DocumentProcessingConfig] = None) -> DocumentProcessor:
    """Factory function per creare un processore di documenti (Docling)."""
    if config is None:
        config = DocumentProcessingConfig()
    return DocumentProcessor(config)
