from docling.document_converter import DocumentConverter

try:
    converter = DocumentConverter()
    print("✅ Docling inizializzato correttamente")
except Exception as e:
    print("❌ Errore Docling all'inizializzazione:", e)
