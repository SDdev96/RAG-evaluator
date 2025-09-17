"""
Document Processing usando PyPDF2 per convertire PDF in formato strutturato Markdown
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import PyPDF2
from config.config import DocumentProcessingConfig


@dataclass
class ProcessedDocument:
    """Rappresenta un documento processato"""
    content: str
    metadata: Dict[str, Any]
    source_path: str
    output_path: str


class DocumentProcessor:
    """Processore di documenti usando PyPDF2"""
    
    def __init__(self, config: DocumentProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("Uso di PyPDF2 per il processing dei PDF")
        
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
        
        self.logger.info(f"Processando documento: {file_path}")
        
        # Usa sempre PyPDF2 o lettura testo semplice
        return self._process_with_fallback(file_path)
    
    def _process_with_fallback(self, file_path: Path) -> ProcessedDocument:
        """Processa il documento usando metodi di fallback"""
        if file_path.suffix.lower() == '.pdf':
            return self._process_pdf_with_pypdf2(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            return self._process_text_file(file_path)
        else:
            raise ValueError(f"Formato non supportato per fallback: {file_path.suffix}")
    
    def _process_pdf_with_pypdf2(self, file_path: Path) -> ProcessedDocument:
        """Processa PDF usando PyPDF2"""
        try:
            content = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    content += f"\n\n## Pagina {page_num + 1}\n\n{page_text}"
            
            metadata = {
                "title": file_path.stem,
                "pages": len(pdf_reader.pages),
                "processing_method": "pypdf2",
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
            self.logger.error(f"Errore nel processing PDF con PyPDF2: {e}")
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
        
        self.logger.info(f"Processati {len(processed_docs)} documenti dalla directory {directory}")
        return processed_docs


def create_processor(config: Optional[DocumentProcessingConfig] = None) -> DoclingProcessor:
    """Factory function per creare un processore di documenti"""
    if config is None:
        config = DocumentProcessingConfig()
    
    return DoclingProcessor(config)
