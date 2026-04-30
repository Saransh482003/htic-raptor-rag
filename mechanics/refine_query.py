import re
import json
from typing import Dict, List, Set, Tuple


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation except hyphens (retain device names like image-free)."""
    text = text.lower().strip()
    # keep hyphens, remove other punctuation
    text = re.sub(r"[^\w\s\-]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    return text.split()


def _default_stopwords() -> Set[str]:
    # General fillers + interrogatives + auxiliaries; keep negations out of this set
    return {
        'what','which','who','whom','whose','when','where','why','how',
        'is','am','are','was','were','be','being','been','do','does','did','done',
        'the','a','an','of','to','in','on','at','for','from','by','with','as','into','about','via','and','or',
        'this','that','these','those','it','its','they','their','them','we','our','you','your','i','me','my',
        'please','kindly','can','could','would','should','will','shall','may','might','let','tell','explain','give','show'
    }


def _domain_keepwords() -> Set[str]:
    # Device/product + core biomedical terms frequently used in this repo
    return {
        'artsens','sphygmocor','arteriograph','ultrasound','tonometry','oscillometric',
        'carotid','brachial','aortic','aorta','radial','arterial','central','peripheral',
        'stiffness','compliance','waveform','calibration','validation','guideline','manual',
        'pressure','blood','cuff','transducer','augmentation','index','velocity','wave','pulse',
        'ai','aix','pwv','bp','pwa','pwg'
    }


def _negations_and_modifiers() -> Set[str]:
    return {'not','no','without','vs','versus','difference','compare','comparison','latest','recent','updated','news','2024','2025'}


def _acronym_expansions() -> Dict[str, List[str]]:
    return {
        'pwv': ['pulse', 'wave', 'velocity'],
        'aix': ['augmentation', 'index'],
        'bp': ['blood', 'pressure'],
        'pwa': ['pulse', 'wave', 'analysis']
    }


def _synonym_expansions() -> Dict[str, List[str]]:
    return {
        'functionalities': ['features', 'capabilities'],
        'functionality': ['features', 'capabilities'],
        'features': ['capabilities', 'functionalities'],
        'precautions': ['warnings', 'safety', 'contraindications'],
        'precaution': ['warning', 'safety', 'contraindication'],
        'usage': ['operation', 'use', 'operational'],
        'use': ['usage', 'operation'],
        'accuracy': ['precision', 'reliability'],
        'setup': ['calibration', 'configuration'],
        'calibration': ['setup', 'configuration'],
        'validation': ['verification', 'benchmark'],
    }


def _device_related_boosts(tokens: List[str]) -> List[str]:
    # If device mentioned, add typical metrics measured by these systems to improve recall
    toks = set(tokens)
    boosts: List[str] = []
    if 'artsens' in toks or 'sphygmocor' in toks or 'arteriograph' in toks:
        boosts += ['pulse', 'wave', 'velocity', 'augmentation', 'index', 'central', 'blood', 'pressure', 'arterial', 'stiffness']
    return boosts


def refine_query(raw: str, enable_expansion: bool = True, min_tokens_after_refine: int = 2) -> Dict[str, object]:
    """
    Heuristically refine a user query for embedding-based retrieval.

    - Removes generic filler while preserving domain terms, negations, and temporal cues.
    - Optionally expands with acronyms/synonyms to improve recall.

    Returns a dict:
      {
        'original': <raw>,
        'reduced': <denoised string>,
        'expanded': <expanded string or reduced>,
        'tokens': [kept tokens],
        'dropped': [removed tokens],
        'expansions_added': [tokens added during expansion]
      }
    """
    original = raw or ""
    norm = _normalize(original)
    toks = _tokenize(norm)

    stop = _default_stopwords()
    keep_domain = _domain_keepwords()
    keep_special = _negations_and_modifiers()

    kept: List[str] = []
    dropped: List[str] = []

    for t in toks:
        if not t:
            continue
        # Keep years like 2023-2026
        if re.fullmatch(r"20\d{2}", t):
            kept.append(t)
            continue
        if t in keep_special or t in keep_domain:
            kept.append(t)
            continue
        if t in stop:
            dropped.append(t)
            continue
        # Short generic tokens of 1 char often noise (except 'ai' handled above)
        if len(t) == 1 and t not in {'x'}:
            dropped.append(t)
            continue
        kept.append(t)

    # If we over-filtered, fallback to normalized original
    if len(kept) < min_tokens_after_refine:
        kept = toks[:]  # use normalized tokens
        dropped = []

    # Deduplicate while preserving order
    seen = set()
    kept = [x for x in kept if not (x in seen or seen.add(x))]

    # Build reduced string
    reduced = " ".join(kept)

    expansions_added: List[str] = []
    if enable_expansion:
        expanded_tokens = kept[:]

        # Acronym expansions (add tokens, not replacing)
        acro = _acronym_expansions()
        for k, phrase_tokens in acro.items():
            if k in kept:
                for pt in phrase_tokens:
                    if pt not in expanded_tokens:
                        expanded_tokens.append(pt)
                        expansions_added.append(pt)

        # Synonym expansions
        syns = _synonym_expansions()
        for t in kept:
            if t in syns:
                for s in syns[t]:
                    # split multiword synonym into tokens and add
                    for st in _tokenize(_normalize(s)):
                        if st not in expanded_tokens:
                            expanded_tokens.append(st)
                            expansions_added.append(st)

        # Device-related default boosts
        for b in _device_related_boosts(kept):
            if b not in expanded_tokens:
                expanded_tokens.append(b)
                expansions_added.append(b)

        # Dedupe preserve order
        seen2 = set()
        expanded_tokens = [x for x in expanded_tokens if not (x in seen2 or seen2.add(x))]
        expanded = " ".join(expanded_tokens)
    else:
        expanded = reduced

    return {
        'original': original,
        'reduced': reduced,
        'expanded': expanded,
        'tokens': kept,
        'dropped': dropped,
        'expansions_added': expansions_added,
    }
