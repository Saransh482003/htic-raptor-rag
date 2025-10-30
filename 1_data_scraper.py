import os
import re
import fitz
import json
from pathlib import Path
from unidecode import unidecode
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from docx import Document
import markdown

load_dotenv()

def clean_text(text: str) -> str:
    """Aggressively clean input text at ingestion time to remove citation noise and
    formatting artifacts, producing clean, LLM-friendly content before chunking.

    Steps:
    - ASCII fold and lowercase
    - Remove control chars and bullets/soft hyphens
    - Undo line-break hyphenation
    - Strip numeric citations [1], [1-3], [2, 5] and numeric-only (2019)
    - Strip bracketed 'et al', 'doi:', or URLs
    - Normalize spaced hyphens in compounds (incl. single letters: a - mode -> a-mode)
    - Fix spaced decimals: 2. 71 -> 2.71
    - Tighten spaces around punctuation and parentheses
    - Collapse whitespace/newlines
    """
    if not isinstance(text, str):
        text = str(text or "")

    t = unidecode(text).lower()

    # Remove control chars
    t = re.sub(r'[\u0000-\u001f\u007f]', ' ', t)

    # Replace common unicode dashes/bullets/soft hyphens
    t = t.replace('\u2014', '-')  # em-dash
    t = t.replace('\u2013', '-')  # en-dash
    t = t.replace('\u2212', '-')  # minus sign
    t = t.replace('\u00ad', '')   # soft hyphen
    t = re.sub(r'[•·●◦∙]', ' ', t)

    # Undo line-break hyphenation (word-\nword -> wordword)
    t = re.sub(r'-\s*\n\s*', '', t)

    # Drop raw URLs/emails
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'\b[\w\.-]+@[\w\.-]+\.[a-z]{2,}\b', ' ', t)

    # Remove numeric citation brackets like [1], [1-3], [1, 2, 3]
    t = re.sub(r"\[\s*(?:\d{1,4}\s*(?:[-–,;]\s*\d{1,4}\s*)*)\]", "", t)
    # Remove multiple adjacent citation brackets e.g., ][
    t = re.sub(r"\]\s*\[", " ", t)

    # Remove numeric-only parentheticals like (2019), (1-3), (1, 2)
    t = re.sub(r"\(\s*(?:\d{1,4}\s*(?:[-–,;]\s*\d{1,4}\s*)*)\)", "", t)

    # Remove obvious citation-like brackets with 'et al', 'doi', or URLs
    t = re.sub(r"\[[^\]]*?et\s+al\.[^\]]*\]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[[^\]]*?doi:\s*[^\]]*\]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[[^\]]*?https?://[^\]]*\]", "", t, flags=re.IGNORECASE)

    # Normalize spaced hyphens in compounds (token - token -> token-token)
    t = re.sub(r"(?i)\b([a-z0-9]+)\s*-\s*([a-z0-9]+)\b", r"\1-\2", t)

    # Fix spaced decimals (e.g., 2. 71 -> 2.71, 4 . 82 -> 4.82)
    t = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", t)

    # Tighten spaces around parentheses
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)

    # Remove stray empty brackets
    t = re.sub(r"\[\s*\]", "", t)

    # Remove spaces before punctuation; ensure single space after , ; : when followed by a word
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,;:])(\S)", r"\1 \2", t)

    # Remove common artifacts like long dotted leaders (TOC) possibly followed by digits
    t = re.sub(r"(\.{2,}|_{2,}|-{3,})\s*\d*", " ", t)

    # Collapse whitespace and normalize newlines
    t = re.sub(r"[ \t\f\v]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)

    return t.strip()


def extract_pdf(pdf_path: str):
    """Extract entire text from a PDF and return a single cleaned block ready for embeddings."""
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        try:
            texts.append(page.get_text("text") or "")
        except Exception:
            continue
    doc.close()

    full_text = "\n".join(texts)
    cleaned = clean_text(full_text)
    pages = [{"page": 1, "text": cleaned, "source": Path(pdf_path).name}]

    os.makedirs(os.getenv("EXTRACTED_DATA_PATH"), exist_ok=True)
    out_path = os.path.join(os.getenv("EXTRACTED_DATA_PATH"), f"{Path(pdf_path).stem}_extracted.json")
    with open(out_path, "w", encoding="utf-16") as f:
        json.dump(pages, f, indent=4, ensure_ascii=False)
    return pages



def extract_txt(txt_path: str):
    """Extract entire text from a TXT file as a single cleaned block."""
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    cleaned = clean_text(content)
    pages = [{"page": 1, "text": cleaned, "source": Path(txt_path).name}]

    os.makedirs(os.getenv("EXTRACTED_DATA_PATH"), exist_ok=True)
    out_path = os.path.join(os.getenv("EXTRACTED_DATA_PATH"), f"{Path(txt_path).stem}_extracted.json")
    with open(out_path, "w", encoding="utf-16") as f:
        json.dump(pages, f, indent=4, ensure_ascii=False)
    return pages


def extract_md(md_path: str):
    """Extract entire text from a Markdown file as a single cleaned block."""
    with open(md_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    html_content = markdown.markdown(content)
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ')
    cleaned = clean_text(text)
    pages = [{"page": 1, "text": cleaned, "source": Path(md_path).name}]

    os.makedirs(os.getenv("EXTRACTED_DATA_PATH"), exist_ok=True)
    out_path = os.path.join(os.getenv("EXTRACTED_DATA_PATH"), f"{Path(md_path).stem}_extracted.json")
    with open(out_path, "w", encoding="utf-16") as f:
        json.dump(pages, f, indent=4, ensure_ascii=False)
    return pages


def extract_html(html_path: str):
    """Extract entire text from HTML/HTM as a single cleaned block."""
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'html.parser')
    # Remove non-content elements
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()

    text = soup.get_text(separator=' ')
    cleaned = clean_text(text)
    pages = [{"page": 1, "text": cleaned, "source": Path(html_path).name}]

    os.makedirs(os.getenv("EXTRACTED_DATA_PATH"), exist_ok=True)
    out_path = os.path.join(os.getenv("EXTRACTED_DATA_PATH"), f"{Path(html_path).stem}_extracted.json")
    with open(out_path, "w", encoding="utf-16") as f:
        json.dump(pages, f, indent=4, ensure_ascii=False)
    return pages


def extract_docx(docx_path: str):
    """Extract entire text from a DOCX file as a single cleaned block."""
    doc = Document(docx_path)
    paragraphs = [p.text for p in doc.paragraphs if p and p.text]
    full_text = "\n".join(paragraphs)
    cleaned = clean_text(full_text)
    pages = [{"page": 1, "text": cleaned, "source": Path(docx_path).name}]

    os.makedirs(os.getenv("EXTRACTED_DATA_PATH"), exist_ok=True)
    out_path = os.path.join(os.getenv("EXTRACTED_DATA_PATH"), f"{Path(docx_path).stem}_extracted.json")
    with open(out_path, "w", encoding="utf-16") as f:
        json.dump(pages, f, indent=4, ensure_ascii=False)
    return pages


def extract_file(file_path: str):
    """
        Main extraction function that routes to appropriate extractor based on file extension.
        Supported formats: PDF, TXT, MD, HTML, HTM, DOCX
    """
    file_extension = Path(file_path).suffix.lower()
    
    extractors = {
        '.pdf': extract_pdf,
        '.txt': extract_txt,
        '.md': extract_md,
        '.html': extract_html,
        '.htm': extract_html,
        '.docx': extract_docx
    }
    
    if file_extension in extractors:
        print(f"Extracting {file_extension[1:].upper()} file: {Path(file_path).name}")
        return extractors[file_extension](file_path)
    else:
        print(f"⚠️ Unsupported file format: {file_extension} for file {Path(file_path).name}")
        return None


# Process all supported file types
files = os.listdir(os.getenv("RAW_DATA_PATH"))
supported_extensions = ['.pdf', '.txt', '.md', '.html', '.htm', '.docx']
supported_files = [f for f in files if Path(f).suffix.lower() in supported_extensions]

print(f"Found {len(supported_files)} supported files to process:")
for file in supported_files:
    print(f"  - {file}")

print("\n🚀 Starting extraction process...\n")

for file in supported_files:
    file_path = os.path.join(os.getenv("RAW_DATA_PATH"), file)
    try:
        extract_file(file_path)
        print(f"✅ Successfully extracted: {file}\n")
    except Exception as e:
        print(f"❌ Error extracting {file}: {str(e)}\n")

print(f"\n✨ Extraction complete! Processed {len(supported_files)} files.")
print(f"📁 Output location: {os.getenv('EXTRACTED_DATA_PATH')}")