from r6_retrieval_mechs import raptor_retrieve
import ollama
import logging
import time
from datetime import datetime
from typing import List, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('raptor_rag_queries.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "model": "llama3",
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "max_context_length": 8000,
    "default_top_k_root": 2,
    "default_top_k_children": 3
}


def validate_context_length(context_text: str, max_length: int = CONFIG["max_context_length"]) -> str:
    """Validate and truncate context if too long."""
    if len(context_text) <= max_length:
        return context_text
    
    logger.warning(f"Context length ({len(context_text)}) exceeds maximum ({max_length}). Truncating...")
    return context_text[:max_length] + "\n\n[Context truncated due to length limits...]"


def answer_llm(question: str, context: List, show_sources: bool = True) -> str:
    """Generate an answer using hierarchical RAPTOR RAG context and LLM."""
    start_time = time.time()
    
    if not context:
        logger.warning(f"No context found for question: {question[:100]}...")
        return "❌ No relevant context found for your question."
    
    # Format context with source information
    formatted_context = []
    sources = set()
    
    for doc in context:
        content = doc.page_content
        source = doc.metadata.get('source', 'Unknown source')
        chunk_id = doc.metadata.get('id', 'Unknown chunk')
        
        sources.add(source)
        formatted_context.append(f"[Chunk: {chunk_id}]\n{content}")
    
    context_text = "\n\n".join(formatted_context)
    context_text = validate_context_length(context_text)
    
    logger.info(f"Processing question with {len(context)} chunks from {len(sources)} sources")

    prompt = f"""You are an expert biomedical assistant with access to a hierarchical knowledge retrieval system (RAPTOR RAG).

    The provided context comes from a multi-level hierarchical summary tree where:
    - Higher-level summaries contain abstract, general information
    - Lower-level chunks contain specific, detailed information
    - Information has been retrieved based on semantic similarity to your question

    Your task is to synthesize information from these hierarchical levels to provide a comprehensive answer.

    Guidelines:
    - Base your answer STRICTLY on the provided hierarchical context
    - Synthesize information across different abstraction levels when relevant
    - When citing information, reference the source document naturally (e.g., "According to the ARTSENS manual...")
    - Do NOT use numeric citations like [1], [2], etc.
    - If information spans multiple hierarchy levels, explain the relationship clearly
    - Maintain professional, clinical tone suitable for medical professionals
    - If context is insufficient, explicitly state what information is missing
    - Structure your response: Brief introduction → Key findings → Clinical implications → Conclusion
    - Prioritize accuracy over completeness - never fabricate details

    Question:
    {question}

    Hierarchical Context from RAPTOR Retrieval:
    {context_text}

    Provide a comprehensive, well-structured answer based on the hierarchical context above:
    """

    try:
        response = ollama.chat(
            model=CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": CONFIG["temperature"],
                "top_p": CONFIG["top_p"],
                "top_k": CONFIG["top_k"],
                "repeat_penalty": CONFIG["repeat_penalty"],
                "stop": ["Question:", "Context:", "Hierarchical Context:", "Guidelines:"]
            }
        )
        
        # Clean up the response
        answer = response['message']['content'].strip()
        
        # Remove duplicate lines and formatting artifacts
        lines = answer.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines and not line.startswith('###'):
                cleaned_lines.append(line)
                seen_lines.add(line)
        
        final_answer = '\n'.join(cleaned_lines)
        
        # Add source information if requested
        if show_sources and sources:
            source_list = '\n'.join([f"• {source}" for source in sorted(sources)])
            final_answer += f"\n\n**Sources consulted:**\n{source_list}"
        
        # Log successful completion
        elapsed_time = time.time() - start_time
        logger.info(f"Answer generated successfully in {elapsed_time:.2f} seconds")
        
        return final_answer
        
    except Exception as e:
        error_msg = f"❌ Error generating answer: {str(e)}\nPlease check if Ollama is running and the model '{CONFIG['model']}' is available."
        logger.error(f"LLM error: {str(e)}")
        return error_msg
    


def save_query_history(question: str, answer: str, sources: List[str], response_time: float):
    """Save query history to JSON file."""
    try:
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": list(sources),
            "response_time": response_time,
            "model": CONFIG["model"]
        }
        
        # Load existing history
        try:
            with open("query_history.json", "r", encoding="utf-8") as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
        
        # Add new entry
        history.append(history_entry)
        
        # Save updated history
        with open("query_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Failed to save query history: {str(e)}")


def interactive_query():
    """Interactive query interface for RAPTOR RAG system."""
    print("🔬 RAPTOR RAG Biomedical Query System")
    print("=" * 50)
    print("🏥 Specialized for biomedical device documentation")
    print("📚 Using hierarchical retrieval (RAPTOR) for comprehensive answers")
    print("💡 Type 'help' for commands, 'quit' to exit\n")
    
    while True:
        try:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using RAPTOR RAG!")
                break
            
            if query.lower() == 'help':
                print("\n📋 Available commands:")
                print("• Type any biomedical question to get an answer")
                print("• 'config' - Show current configuration")
                print("• 'stats' - Show query statistics")
                print("• 'help' - Show this help message")
                print("• 'quit'/'exit'/'q' - Exit the system")
                continue
            
            if query.lower() == 'config':
                print("\n⚙️ Current Configuration:")
                for key, value in CONFIG.items():
                    print(f"• {key}: {value}")
                continue
            
            if query.lower() == 'stats':
                try:
                    with open("query_history.json", "r", encoding="utf-8") as f:
                        history = json.load(f)
                    print(f"\n📊 Query Statistics:")
                    print(f"• Total queries: {len(history)}")
                    if history:
                        avg_time = sum(entry.get('response_time', 0) for entry in history) / len(history)
                        print(f"• Average response time: {avg_time:.2f} seconds")
                        recent_query = history[-1]
                        print(f"• Last query: {recent_query['timestamp']}")
                except FileNotFoundError:
                    print("\n📊 No query history found yet.")
                continue
            
            if not query:
                print("⚠️  Please enter a valid question.")
                continue
            
            print("🔍 Retrieving relevant information from hierarchical knowledge base...")
            start_time = time.time()
            
            # Retrieve with optimized parameters for biomedical content
            results = raptor_retrieve(query, top_k_root=CONFIG["default_top_k_root"], top_k_children=CONFIG["default_top_k_children"])
            
            if not results:
                print("❌ No relevant information found. Try rephrasing your question.")
                continue
            
            print(f"📚 Found {len(results)} relevant chunks across hierarchy levels")
            print("\n🤖 Generating comprehensive answer...\n")
            
            # Generate answer
            answer = answer_llm(query, results, show_sources=True)
            response_time = time.time() - start_time
            
            # Extract sources for history
            sources = set()
            for doc in results:
                source = doc.metadata.get('source', 'Unknown source')
                sources.add(source)
            
            print("=" * 70)
            print("📋 ANSWER:")
            print("=" * 70)
            print(answer)
            print("=" * 70)
            print(f"⏱️  Response time: {response_time:.2f} seconds")
            
            # Save to history
            save_query_history(query, answer, list(sources), response_time)
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again with a different question.")


def single_query(question: str, top_k_root: Optional[int] = None, top_k_children: Optional[int] = None, show_sources: bool = True) -> str:
    """Process a single query and return the answer."""
    top_k_root = top_k_root or CONFIG["default_top_k_root"]
    top_k_children = top_k_children or CONFIG["default_top_k_children"]
    
    logger.info(f"Processing single query: {question[:100]}...")
    start_time = time.time()
    
    results = raptor_retrieve(question, top_k_root=top_k_root, top_k_children=top_k_children)
    answer = answer_llm(question, results, show_sources=show_sources)
    
    response_time = time.time() - start_time
    
    # Extract sources for history
    sources = set()
    for doc in results:
        source = doc.metadata.get('source', 'Unknown source')
        sources.add(source)
    
    # Save to history
    save_query_history(question, answer, list(sources), response_time)
    
    return answer


if __name__ == "__main__":
    # Run interactive mode by default
    interactive_query()