from ddgs import DDGS
from typing import List, Dict

def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Free search via DuckDuckGo (no API key needed)
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        formatted = []
        for r in results:
            formatted.append({
                'title': r.get('title'),
                'link': r.get('href'),
                'snippet': r.get('body'),
                'source': 'duckduckgo'
            })
        
        return formatted
    
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return []

# Usage
results = duckduckgo_search(input("Enter your search query: "), max_results=10)
with open("duckduckgo_results.json", "w") as f:
    import json
    json.dump(results, f, indent=4)