import fitz

class PDFExtractionError(Exception):
    pass

def extract_pages(file_bytes: bytes) -> list[dict]:
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    
        pages = []
    
        try:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text().strip()
                if text:
                    pages.append({
                        "page": page_number,
                        "text": text,
                    })
        finally:
            document.close()
            
        
        if pages:
            return pages
        
    except Exception:
        pass
    
    try:
        text = file_bytes.decode("utf-8").strip()
        if text:
            return [
                {
                    "page": 1,
                    "text": text,
                }
            ]
            
    except UnicodeDecodeError:
        pass
    
    raise PDFExtractionError("Failed to extract text from PDF.")
    