import os
import re
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# Mandatory tokenizer-aware length
# Use cl100k_base to align with GPT-style tokenization (good fit with your GPT summarizer
# and acceptable for Gemma embeddings when sizing RAG chunks).
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

def _token_length(txt: str) -> int:
    return len(_TOKENIZER.encode(txt or ""))

load_dotenv()

# Configure chunking (interpreted in tokens if tokenizer available)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Rich separators: try strong boundaries first, then weaken as needed
SEPARATORS = [
    "\n\n",  # paragraph
    "\n",    # line
    ". ",    # sentence
    "! ",
    "? ",
    "; ",
    ": ",
    "— ",   # em-dash
    " - ",  # spaced dash
    " ",     # space
    ""       # character fallback
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=_token_length,
    separators=SEPARATORS,
)

def preprocess_text(text: str) -> str:
    """Aggressively clean input text to remove citation noise and formatting artifacts,
    while preserving readable plain sentences for the LLM.

    Heuristics applied:
    - Remove numeric citation brackets like [1], [1-3], [2, 5, 7]
    - Remove numeric-only parentheticals like (2019), (1-3), (1, 2)
    - Remove common citation-like brackets with 'et al', 'doi', or 'http'
    - Normalize spaced hyphens in compounds ("carotid - femoral" -> "carotid-femoral", "a - mode" -> "a-mode")
    - Strip bullet characters and soft hyphens
    - Normalize spacing around punctuation and parentheses
    - Fix spaced decimals like "2. 71" -> "2.71"
    - Collapse excessive whitespace and tidy newlines
    """
    if not text:
        return ""

    t = text

    # Replace common unicode dashes/bullets/soft hyphens
    t = t.replace("\u2014", "-")  # em-dash
    t = t.replace("\u2013", "-")  # en-dash
    t = t.replace("\u2212", "-")  # minus sign
    t = t.replace("\u00ad", "")   # soft hyphen
    t = re.sub(r"[•·●◦∙]", " ", t)

    # Undo line-break hyphenation (word-\nword -> wordword)
    t = re.sub(r"-\s*\n\s*", "", t)

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

    # Normalize spaced hyphens in compounds (token - token -> token-token), allow 1+ chars
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

    # Collapse excessive internal whitespace, normalize newlines
    t = re.sub(r"[ \t\f\v]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)

    return t.strip()

def create_chunks(corpus, file_id):
    chunks, indexer = [], 0
    raw_text = corpus.get("text", "") or ""
    cleaned = preprocess_text(raw_text)
    splits = text_splitter.split_text(cleaned)
    for split in splits:
        text = (split or "").strip()
        # Remove stray leading punctuation/bullets introduced by parsing (., •, -, etc.)
        text = re.sub(r"^[\s\u2022\-•·\.]+", "", text)
        if not text:
            continue
        # Ensure sentence boundary punctuation but do not duplicate
        if text and text[-1] not in ".!?":
            text = text + "."
        chunks.append({
            "id": f"file_{file_id}_chunk_{indexer}_level_0",
            "text": text.lower(),
            "source": corpus["source"],
        })
        indexer += 1
    return chunks


with open(os.path.join(os.getenv("ESSENTIALS_PATH"), "group_chunks.json"), "r", encoding="utf-16") as f:
    doc_corpus = json.loads(f.read())
    chunk_generator = {}
    for file_id, file_content in enumerate(doc_corpus):
        chunks = create_chunks(file_content, file_id)
        chunk_generator[f"file_{file_id}"] = {"levels": {"level_0": chunks}, "source": file_content["source"]}

    with open(os.getenv("HIERARCHY_STORE_PATH"), "w", encoding="utf-16") as out_f:
        json.dump(chunk_generator, out_f, ensure_ascii=False, indent=4)