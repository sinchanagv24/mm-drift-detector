import re
from typing import Dict, Any, List

def extract_pdf_text(path: str) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        texts = []
        for page in doc:
            texts.append(page.get_text())
        return "\n".join(texts)
    except Exception:
        return ""

def ocr_image_text(path: str) -> str:
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

def find_signals(texts: List[str]) -> Dict[str, Any]:
    joined = " ".join([t.lower() for t in texts if t])
    return {
        "currency_eur": bool(re.search(r"\beur\b", joined)),
        "currency_usd": bool(re.search(r"\busd\b", joined)),
        "api_v2": bool(re.search(r"api\s*v2", joined)),
        "migration": bool(re.search(r"migrat", joined)),
        "conversion_note": bool(re.search(r"conversion|convert", joined)),
        "category_change": bool(re.search(r"category|threshold", joined)),
    }
