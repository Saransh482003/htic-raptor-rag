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
    """
        Common text cleaning function for all file types.
        Applies the same cleaning logic as PDF extraction.
    """
    text = unidecode(text)
    text = re.sub(r'[\u0000-\u001F\u007F]', '', text)
    text = text.encode().decode("unicode_escape")
    text = text.replace('\"', '\'')
    text = re.sub(r'\.{5,}\s*\d*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    cleaned_text = ' '.join(text.split())
    return cleaned_text


def extract_pdf(pdf_path: str):
    """
        Extracts text page by page from a PDF.
        Returns: list of {"page": int, "text": str, "source": str}
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

    os.makedirs(os.getenv("EXTRACTED_DATA_PATH"), exist_ok=True)
    with open(f"{os.getenv("EXTRACTED_DATA_PATH")}/{Path(pdf_path).stem}_extracted.json", "w", encoding="utf-16") as f:
        json.dump(pages, f, indent=4)
    return pages



def extract_txt(txt_path: str):
    """
        Extracts text from a plain text file.
        Returns: list of {"page": int, "text": str, "source": str}
        Splits by double newlines to create logical "pages"
    """
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split content into chunks by double newlines (paragraphs)
    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
    
    pages = []
    for i, chunk in enumerate(chunks):
        cleaned_text = clean_text(chunk)
        if cleaned_text:  # Only add non-empty chunks
            pages.append({"page": i + 1, "text": cleaned_text, "source": Path(txt_path).name})
    
    # If no chunks found, treat entire file as one page
    if not pages:
        cleaned_text = clean_text(content)
        pages.append({"page": 1, "text": cleaned_text, "source": Path(txt_path).name})
    
    os.makedirs(os.getenv("EXTRACTED_DATA_PATH"), exist_ok=True)
    with open(f"{os.getenv("EXTRACTED_DATA_PATH")}/{Path(txt_path).stem}_extracted.json", "w", encoding="utf-16") as f:
        json.dump(pages, f, indent=4)
    return pages


def extract_md(md_path: str):
    """
        Extracts text from a Markdown file.
        Returns: list of {"page": int, "text": str, "source": str}
        Splits by headers to create logical "pages"
    """
    with open(md_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Convert markdown to HTML then extract text
    html_content = markdown.markdown(content)
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Split by headers (h1, h2, h3) to create sections
    sections = []
    current_section = []
    
    for element in soup.descendants:
        if element.name in ['h1', 'h2', 'h3']:
            if current_section:
                section_text = ' '.join([str(e) for e in current_section if hasattr(e, 'get_text')])
                sections.append(BeautifulSoup(section_text, 'html.parser').get_text())
                current_section = []
        if hasattr(element, 'get_text'):
            current_section.append(element)
    
    # Add last section
    if current_section:
        section_text = ' '.join([str(e) for e in current_section if hasattr(e, 'get_text')])
        sections.append(BeautifulSoup(section_text, 'html.parser').get_text())
    
    # If no sections found, use entire text
    if not sections:
        sections = [soup.get_text()]
    
    pages = []
    for i, section in enumerate(sections):
        cleaned_text = clean_text(section)
        if cleaned_text:
            pages.append({"page": i + 1, "text": cleaned_text, "source": Path(md_path).name})
    
    os.makedirs(os.getenv("EXTRACTED_DATA_PATH"), exist_ok=True)
    with open(f"{os.getenv("EXTRACTED_DATA_PATH")}/{Path(md_path).stem}_extracted.json", "w", encoding="utf-16") as f:
        json.dump(pages, f, indent=4)
    return pages


def extract_html(html_path: str):
    """
        Extracts text from an HTML or HTM file.
        Returns: list of {"page": int, "text": str, "source": str}
        Splits by major sections or headers to create logical "pages"
    """
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(['script', 'style', 'nav', 'footer', 'header']):
        script.decompose()
    
    # Try to split by major sections
    sections = []
    
    # First try to find sections by div, article, section tags
    major_elements = soup.find_all(['section', 'article', 'div'], class_=re.compile(r'content|main|section|article'))
    
    if major_elements:
        for element in major_elements:
            text = element.get_text()
            if text.strip():
                sections.append(text)
    else:
        # Fall back to splitting by headers
        headers = soup.find_all(['h1', 'h2', 'h3'])
        if headers:
            for i, header in enumerate(headers):
                section_content = []
                for sibling in header.next_siblings:
                    if hasattr(sibling, 'name') and sibling.name in ['h1', 'h2', 'h3']:
                        break
                    if hasattr(sibling, 'get_text'):
                        section_content.append(sibling.get_text())
                section_text = ' '.join(section_content)
                if section_text.strip():
                    sections.append(header.get_text() + ' ' + section_text)
        else:
            # Last resort: use entire text
            sections = [soup.get_text()]
    
    pages = []
    for i, section in enumerate(sections):
        cleaned_text = clean_text(section)
        if cleaned_text:
            pages.append({"page": i + 1, "text": cleaned_text, "source": Path(html_path).name})
    
    os.makedirs(os.getenv("EXTRACTED_DATA_PATH"), exist_ok=True)
    with open(f"{os.getenv("EXTRACTED_DATA_PATH")}/{Path(html_path).stem}_extracted.json", "w", encoding="utf-16") as f:
        json.dump(pages, f, indent=4)
    return pages


def extract_docx(docx_path: str):
    """
        Extracts text from a DOCX file.
        Returns: list of {"page": int, "text": str, "source": str}
        Each paragraph or section becomes a logical "page"
    """
    doc = Document(docx_path)
    
    pages = []
    page_num = 1
    current_text = []
    
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            # Empty paragraph might indicate section break
            if current_text:
                combined_text = ' '.join(current_text)
                cleaned_text = clean_text(combined_text)
                if cleaned_text:
                    pages.append({"page": page_num, "text": cleaned_text, "source": Path(docx_path).name})
                    page_num += 1
                current_text = []
        else:
            current_text.append(text)
            
            # If paragraph style indicates heading, create a new page
            if paragraph.style.name.startswith('Heading'):
                combined_text = ' '.join(current_text)
                cleaned_text = clean_text(combined_text)
                if cleaned_text:
                    pages.append({"page": page_num, "text": cleaned_text, "source": Path(docx_path).name})
                    page_num += 1
                current_text = []
    
    # Add remaining text
    if current_text:
        combined_text = ' '.join(current_text)
        cleaned_text = clean_text(combined_text)
        if cleaned_text:
            pages.append({"page": page_num, "text": cleaned_text, "source": Path(docx_path).name})
    
    os.makedirs(os.getenv("EXTRACTED_DATA_PATH"), exist_ok=True)
    with open(f"{os.getenv("EXTRACTED_DATA_PATH")}/{Path(docx_path).stem}_extracted.json", "w", encoding="utf-16") as f:
        json.dump(pages, f, indent=4)
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