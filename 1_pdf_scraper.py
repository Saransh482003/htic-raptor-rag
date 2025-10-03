import os
import re
import fitz
import json
from pathlib import Path
from unidecode import unidecode

def extract_text_by_page(pdf_path: str):
    """
        Extracts text page by page from a PDF.
        Returns: list of {"page": int, "text": str}
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = unidecode(text)
        text = re.sub(r'[\u0000-\u001F\u007F]', '', text)
        text = text.encode().decode("unicode_escape")
        text = text.replace('\"', '\'')
        text = re.sub(r'\.{5,}\s*\d*', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        cleaned_text = ' '.join(text.split())
        pages.append({"page": i + 1, "text": cleaned_text, "source": Path(pdf_path).name})
    doc.close()

    os.makedirs("./data/extracted_pdfs", exist_ok=True)
    with open(f"./data/extracted_pdfs/{Path(pdf_path).stem}_extracted.json", "w", encoding="utf-16") as f:
        json.dump(pages, f, indent=4)
    return pages


files = os.listdir("./data/raw_pdfs")
pdf_files = [f for f in files if f.endswith(".pdf")]

for pdf_file in pdf_files:
    extract_text_by_page(os.path.join("./data/raw_pdfs", pdf_file))