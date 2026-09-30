import fitz

class PDFExtractionError(Exception):
    pass

def extract_text(file_bytes: bytes) -> str:
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    
        pages = []
    
        try:
            for page in document:
                text = page.get_text()
                if text:
                    pages.append(text)
        finally:
            document.close()
            
        text = "\n".join(pages).strip()
        
        if text:
            return text
    except Exception:
        pass
    
    try:
        text = file_bytes.decode("utf-8").strip()
        if text:
            return text
    except UnicodeDecodeError:
        pass
    
    raise PDFExtractionError("Failed to extract text from PDF.")
    