import json
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import os
from ddgs import DDGS
import trafilatura
import time
from typing import List, Dict, Optional
from urllib.parse import urlparse
from mechanics.refine_query import refine_query

load_dotenv()

# Load vectorstore
embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDINGS_MODEL"), base_url="http://127.0.0.1:11434")
print(os.getenv("CHROMA_PERSIST_PATH", "./chromaDB_store"))
vectorstore = Chroma(
    persist_directory=os.getenv("CHROMA_PERSIST_PATH", "./chromaDB_store"),
    embedding_function=embeddings
)
# Load original summary tree (for hierarchy reference if needed)
with open(os.getenv("HIERARCHY_STORE_PATH"), "r", encoding="utf-16") as f:
    summary_tree = json.load(f)

MAX_RETRIEVED_CHUNKS = int(os.getenv("MAX_RETRIEVED_CHUNKS", "5"))


# ========== HELPER FUNCTIONS ==========

def _add_unique(results, collector, seen_ids):
    """Helper to append only unseen documents by chunk id."""
    for doc in results or []:
        doc_id = doc.metadata.get("id") if hasattr(doc, "metadata") else None
        if not doc_id:
            doc_id = f"pc:{hash(getattr(doc, 'page_content', ''))}"
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        collector.append(doc)


def _vector_search_with_scores(query: str, k: int, *, filter: Optional[dict] = None) -> List[Document]:
    """Run vector search and annotate documents with similarity scores."""
    if k <= 0:
        return []

    try:
        raw_results = vectorstore.similarity_search_with_relevance_scores(query, k=k, filter=filter)
    except Exception:
        # Fallback to basic search without explicit scores
        fallback_docs = vectorstore.similarity_search(query, k=k, filter=filter)
        raw_results = [(doc, None) for doc in fallback_docs]

    docs = []
    total = max(len(raw_results), 1)
    for rank, (doc, score) in enumerate(raw_results):
        metadata = dict(doc.metadata) if doc.metadata else {}
        if score is None:
            # Derive a monotonic proxy score to preserve ordering
            score = 1.0 - (rank / total)
        metadata["similarity_score"] = float(score)
        doc.metadata = metadata
        docs.append(doc)
    return docs


def _keyword_rescue_root_search(query_tokens: List[str], summary_tree: Dict) -> List[Document]:
    """
    Scans root-level summaries for exact keyword matches.
    This acts as a safety net for rare proper nouns (e.g. 'GymLink') 
    that embeddings might miss.
    """
    rescued_docs = []
    if not query_tokens:
        return rescued_docs

    # 1. Filter tokens to avoid noise
    # We must ignore common stopwords even if they passed through refinement fallback
    stopwords = {
        'what','which','who','whom','whose','when','where','why','how',
        'is','am','are','was','were','be','being','been','do','does','did','done',
        'the','a','an','of','to','in','on','at','for','from','by','with','as','into','about','via','and','or',
        'this','that','these','those','it','its','they','their','them','we','our','you','your','i','me','my',
        'please','kindly','can','could','would','should','will','shall','may','might'
    }
    # Also filter very short tokens (1-2 chars) unless they are specific codes
    effective_tokens = [t for t in query_tokens if t.lower() not in stopwords and len(t) > 2]
    
    if not effective_tokens:
        return rescued_docs
        
    print(f"   🚑 Rescue scanning for: {effective_tokens}")

    # Flatten root nodes for scanning
    # We only scan the highest level available for each file
    for file_id, file_data in summary_tree.items():
        levels = file_data.get("levels", {})
        if not levels:
            continue
        
        # Find max level (root)
        # Level keys are like "level_0", "level_1". We want the highest number.
        try:
            max_level_key = max(levels.keys(), key=lambda x: int(x.split("_")[1]))
            root_chunks = levels[max_level_key]
        except ValueError:
            continue

        for chunk in root_chunks:
            text = chunk.get("text", "").lower()
            # Check if ANY significant rare token appears.
            # We assume query_tokens are already filtered/refined keywords.
            
            matches = sum(1 for token in effective_tokens if token in text)
            
            if matches > 0:
                # Create a Document object to match vectorstore output
                rescue_score = matches / max(len(effective_tokens), 1)
                doc = Document(
                    page_content=chunk.get("text", ""),
                    metadata={
                        "id": chunk.get("id"),
                        "source": file_data.get("source", ""),
                        "level": max_level_key,
                        "chunk_source": json.dumps(chunk.get("source", [])), # Maintain lineage
                        "similarity_score": rescue_score,
                        "keyword_rescue_matches": matches
                    }
                )
                rescued_docs.append(doc)
    
    return rescued_docs


def _hierarchical_traversal(query: str, summary_tree: dict, top_k_root: int, top_k_children: int, 
                          refined_tokens: List[str] = None, enable_keyword_rescue: bool = True) -> List[Document]:
    """Perform hierarchical RAPTOR tree traversal."""
    # Collect root (max-level) chunk IDs
    root_ids = []
    max_levels = set()
    for file_id, file_data in summary_tree.items():
        levels = list(file_data["levels"].keys())
        if not levels:
            continue
        max_level_key = max(levels, key=lambda x: int(x.split("_")[1]))
        max_levels.add(max_level_key)
        for chunk in file_data["levels"][max_level_key]:
            root_ids.append(chunk["id"])

    # 1. Vector Search for Roots
    root_results = []
    if root_ids:
        root_results = _vector_search_with_scores(
            query,
            k=min(top_k_root, len(root_ids)),
            filter={"id": {"$in": root_ids}}
        )

    # Fallback: filter by level if ID filter returns nothing
    if not root_results and max_levels:
        root_results = _vector_search_with_scores(
            query,
            k=top_k_root,
            filter={"level": {"$in": list(max_levels)}}
        )

    # 2. Keyword Rescue for Roots (Safety Net)
    if enable_keyword_rescue and refined_tokens:
        # Check which tokens are already covered by vector search results
        retrieved_text = " ".join([d.page_content.lower() for d in root_results])
        
        missing_tokens = []
        for token in refined_tokens:
            # Only rescue tokens that are NOT present in the vector search results
            if token.lower() not in retrieved_text:
                missing_tokens.append(token)
        
        if missing_tokens:
            rescued_docs = _keyword_rescue_root_search(missing_tokens, summary_tree)
            if rescued_docs:
                print(f"🚑 Keyword Rescue: Found {len(rescued_docs)} documents matching missing tokens: {missing_tokens}")
                # Merge rescued docs into root_results, avoiding duplicates
                existing_ids = {d.metadata.get("id") for d in root_results}
                for d in rescued_docs:
                    if d.metadata.get("id") not in existing_ids:
                        root_results.append(d)
        else:
             print("✓ Keyword Rescue: All tokens found in vector results, skipping rescue.")

    # Store all retrieved documents from all levels
    all_results = []
    seen_ids = set()
    _add_unique(root_results, all_results, seen_ids)

    def descend(children_ids):
        if not children_ids:
            return
        unique_children_ids = list(dict.fromkeys(children_ids))
        k = min(top_k_children, len(unique_children_ids))
        
        child_results = _vector_search_with_scores(
            query,
            k=k,
            filter={"id": {"$in": unique_children_ids}}
        )
        _add_unique(child_results, all_results, seen_ids)

        next_children_ids = []
        for doc in child_results:
            src_raw = doc.metadata.get("chunk_source")
            try:
                src_list = json.loads(src_raw) if isinstance(src_raw, str) else (src_raw or [])
            except Exception:
                src_list = []
            if isinstance(src_list, list):
                next_children_ids.extend(src_list)

        if not next_children_ids:
            if k < len(unique_children_ids):
                leaf_all = _vector_search_with_scores(
                    query,
                    k=len(unique_children_ids),
                    filter={"id": {"$in": unique_children_ids}}
                )
                _add_unique(leaf_all, all_results, seen_ids)
            return

        descend(list(dict.fromkeys(next_children_ids)))

    children = []
    for doc in root_results:
        src_raw = doc.metadata.get("chunk_source")
        try:
            src_list = json.loads(src_raw) if isinstance(src_raw, str) else (src_raw or [])
        except Exception:
            src_list = []
        if isinstance(src_list, list):
            children.extend(src_list)
    
    descend(list(dict.fromkeys(children)))
    return all_results


def _should_trigger_web_search(query: str, raptor_results: List, min_results_threshold: int = 3) -> bool:
    """Determine if web search is necessary."""
    temporal_keywords = [
        'latest', 'recent', 'current', 'new', 'updated', 'today', 'now',
        '2024', '2025', 'this year', 'this month', 'breaking', 'news'
    ]
    
    query_lower = query.lower()
    has_temporal_keyword = any(keyword in query_lower for keyword in temporal_keywords)
    has_insufficient_results = len(raptor_results) < min_results_threshold
    
    if has_insufficient_results:
        print(f"🌐 Web search triggered: Insufficient RAPTOR results ({len(raptor_results)} < {min_results_threshold})")
        return True
    
    if has_temporal_keyword:
        print(f"🌐 Web search triggered: Temporal query detected")
        return True
    
    return False


def _web_search_snippets(query: str, max_results: int = 3) -> List[Dict]:
    """Perform DuckDuckGo search and return snippets."""
    try:
        print(f"🔍 Searching web for: {query}")
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))
        
        if not search_results:
            print("⚠️ No web search results found")
            return []
        
        print(f"✅ Found {len(search_results)} web results")
        
        extracted_results = []
        for i, result in enumerate(search_results, 1):
            url = result.get('href', '')
            title = result.get('title', '')
            snippet = result.get('body', '')
            
            extracted_results.append({
                'title': title,
                'url': url,
                'snippet': snippet,
                'full_body': snippet,
                'body_length': len(snippet),
                'source': 'web_search'
            })
            print(f"   {i}. {title[:60]}... ({len(snippet)} chars)")
        
        return extracted_results
    
    except Exception as e:
        print(f"❌ Web search error: {e}")
        return []


# ========== MAIN RETRIEVAL FUNCTION ==========

def raptor_retrieve(query: str, summary_tree: dict, top_k_root: int = 1, top_k_children: int = 2, use_query_refinement: bool = False, use_web_search: bool = False, max_web_results: int = 3, enable_keyword_rescue: bool = False) -> dict:
    """
    Unified RAPTOR retrieval with optional query refinement and web search.
    
    Args:
        query: User's search query
        summary_tree: RAPTOR summary tree
        top_k_root: Top-k for root level retrieval
        top_k_children: Top-k for children level retrieval
        use_query_refinement: Enable query refinement for better embeddings (default: True)
        use_web_search: Enable conditional web search fallback (default: False)
        max_web_results: Maximum number of web results to fetch
        enable_keyword_rescue: Enable keyword rescue for rare terms (default: True)
    
    Returns:
        Dict containing:
            - 'raptor_results': List of RAPTOR documents
            - 'web_results': List of web search results (if triggered)
            - 'all_contexts': Combined context for LLM (dicts with content/metadata/source_type)
            - 'sources': Source attribution info
            - 'query_info': Original and refined query details
    """
    print(f"\n{'='*80}")
    print(f"🔍 RAPTOR RETRIEVAL: {query}")
    print(f"{'='*80}\n")
    
    # Step 1: Query refinement (optional)
    original_query = query
    retrieval_query = query.lower()
    query_info = {'original': original_query, 'refined': None, 'used_refinement': use_query_refinement}
    refined_tokens = []

    if use_query_refinement:
        refined = refine_query(query, enable_expansion=False)
        retrieval_query = refined['reduced']
        refined_tokens = refined['tokens'] # Get tokens for keyword rescue
        query_info['refined'] = retrieval_query
        query_info['tokens_kept'] = refined['tokens']
        query_info['tokens_dropped'] = refined['dropped']
        print(f"🔧 Query refinement: '{original_query}' → '{retrieval_query}'\n")
    else:
        # If refinement is off, we can still try to use simple tokens for rescue
        # But better to rely on the user enabling refinement if they want this feature.
        # Or we can just split the query.
        refined_tokens = query.lower().split()
    
    # Step 2: RAPTOR hierarchical retrieval
    print("📚 Retrieving from RAPTOR tree...")
    # Pass refined_tokens to traversal for keyword rescue
    raptor_results = _hierarchical_traversal(
        retrieval_query, 
        summary_tree, 
        top_k_root, 
        top_k_children, 
        refined_tokens=refined_tokens,
        enable_keyword_rescue=enable_keyword_rescue
    )
    print(f"✓ Found {len(raptor_results)} RAPTOR documents\n")
    
    # Step 3: Conditional web search
    web_results = []
    if use_web_search and _should_trigger_web_search(original_query, raptor_results):
        web_results = _web_search_snippets(original_query, max_results=max_web_results)
        print()
    elif use_web_search:
        print("✓ RAPTOR results sufficient, skipping web search\n")
    
    # Step 4: Combine contexts (web results first for temporal queries)
    sources = {'raptor': len(raptor_results), 'web': len(web_results)}

    web_contexts = []
    for web_doc in web_results:
        web_contexts.append({
            'content': web_doc['full_body'],
            'metadata': {
                'title': web_doc['title'],
                'url': web_doc['url'],
                'source': web_doc['url']
            },
            'source_type': 'web',
            'source': web_doc['url'],
            'score': 1e6  # Always prioritize web snippets when present
        })

    raptor_contexts = []
    for doc in raptor_results:
        similarity_score = float(doc.metadata.get('similarity_score', 0.0))
        raptor_contexts.append({
            'content': doc.page_content,
            'metadata': doc.metadata,
            'source_type': 'raptor',
            'source': doc.metadata.get('source', 'unknown'),
            'score': similarity_score
        })

    raptor_contexts.sort(key=lambda ctx: ctx.get('score', 0.0), reverse=True)
    all_contexts = web_contexts + raptor_contexts
    limited_contexts = all_contexts[:MAX_RETRIEVED_CHUNKS]

    print(f"📊 RETRIEVAL SUMMARY:")
    print(f"   RAPTOR: {sources['raptor']} documents")
    print(f"   Web: {sources['web']} documents")
    print(f"   Total Combined: {len(all_contexts)} contexts")
    if len(all_contexts) > MAX_RETRIEVED_CHUNKS:
        print(f"   ➜ Truncated to top {MAX_RETRIEVED_CHUNKS} contexts after ranking")
    print(f"{'='*80}\n")
    
    return {
        'raptor_results': raptor_results,
        'web_results': web_results,
        'all_contexts': limited_contexts,
        'sources': sources,
        'query_info': query_info
    }


# question = "Under what specific surgical history condition should the Arteriograph cuff NOT be placed on a patient's arm?"
# result = raptor_retrieve(
#     question, 
#     summary_tree, 
#     top_k_root=5, 
#     top_k_children=2,
#     use_query_refinement=True,
#     use_web_search=True,
#     max_web_results=3,
#     enable_keyword_rescue=True
# )
# print("--- Retrieved Content ---")
# for i, context in enumerate(result['all_contexts']):
#     print(f"\nChunk {i+1}:")
#     print(context['content'])