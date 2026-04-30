from mechanics.retrieval_mechs import raptor_retrieve
import ollama
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from dotenv import load_dotenv
import os
from ollama import Client
from rich.console import Console
from rich.markdown import Markdown
from mechanics.ollama_usage_tracker import usage_tracker
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

console = Console()

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

CONFIG = json.loads(os.getenv("OLLAMA_LLM_RESPONSE_CONFIG"))
MAX_CONTEXTS = int(os.getenv("MAX_RETRIEVED_CHUNKS", "5"))

with open(os.getenv("HIERARCHY_STORE_PATH"), "r", encoding="utf-16") as f:
    summary_tree = json.load(f)

def validate_context_length(context_text: str, max_length: int = CONFIG["max_context_length"]) -> str:
    """Validate and truncate context if too long."""
    if len(context_text) <= max_length:
        return context_text
    
    logger.warning(f"Context length ({len(context_text)}) exceeds maximum ({max_length}). Truncating...")
    return context_text[:max_length] + "\n\n[Context truncated due to length limits...]"


def answer_llm(question: str, context: List, show_sources: bool = True, web_results: List = None, long_answer: bool = False) -> str:
    """Generate an answer using hierarchical RAPTOR RAG context, web sources, and LLM."""
    pipeline_start = time.time()
    
    if not context:
        logger.warning(f"No context found for question: {question[:100]}...")
        return "❌ No relevant context found for your question."
    
    # Format context with source information
    formatted_context = []
    sources = []  # Changed to list to keep all chunks, not just unique sources
    
    # Process RAPTOR results
    for i, doc in enumerate(context):
        # Check if this is a dict (from hybrid_retrieve) or Document object (from raptor_retrieve)
        if isinstance(doc, dict):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            source_type = doc.get('source_type', 'raptor')
        else:
            content = doc.page_content
            metadata = doc.metadata
            source_type = 'raptor'
        
        if source_type == 'web':
            # Web source formatting
            title = metadata.get('title', 'Web Source')
            url = metadata.get('url', '')
            chunk_info = {
                'file': f"🌐 Web: {title}",
                'chunk_id': url,
                'content': content[:500] + "..." if len(content) > 500 else content
            }
            sources.append(chunk_info)
            formatted_context.append(f"[Web Source: {title}]\nURL: {url}\n{content}")
        else:
            # RAPTOR source formatting
            source = metadata.get('file_id', 'Unknown source')
            chunk_id = metadata.get('id', 'Unknown chunk')
            level = metadata.get('level', 'Unknown level')
            
            # Add more descriptive source information
            if source != 'Unknown source':
                # Clean up the source name for better readability
                clean_source = source.replace('_manual_extracted', ' Manual').replace('_', ' ').title()
                # Store file, chunk ID, and content for sources section
                chunk_info = {
                    'file': clean_source,
                    'chunk_id': chunk_id,
                    'content': content[:500] + "..." if len(content) > 500 else content  # Truncate long content
                }
                sources.append(chunk_info)
                formatted_context.append(f"[Source: {clean_source} | Level: {level} | Chunk: {chunk_id}]\n{content}")
            else:
                chunk_info = {
                    'file': 'Unknown source',
                    'chunk_id': chunk_id,
                    'content': content[:500] + "..." if len(content) > 500 else content
                }
                sources.append(chunk_info)
                formatted_context.append(f"[Chunk: {chunk_id}]\n{content}")
    
    context_text = "\n\n".join(formatted_context)
    # context_text = validate_context_length(context_text)
    # print("\n\n\n\n\n")
    # print(context_text)
    logger.info(f"Processing question with {len(context)} chunks from {len(sources)} sources")


    if long_answer:
        output_rules = """
            OUTPUT FORMAT (LONG ANSWER):
            - Structure strictly as:
            Brief introduction →
            Key findings →
            Clinical implications →
            Conclusion
            - Provide thorough, well-explained synthesis.
            - Maintain a professional clinical tone.
        """
    else:
        output_rules = """
            OUTPUT FORMAT (SHORT ANSWER — HARD CONSTRAINT):
            - Output NO MORE THAN 2–3 sentences total.
            - Mention all numeric details.
            - Use concise, telegraphic wording.
            - No explanations, transitions, or filler.
            - Summarize aggressively.
            - If more detail is required, omit it.
            - If your draft exceeds 3 sentences, TRUNCATE it.
            - NO NEW LINE CHARACTERS, NO MARKDOWN FORMATING. Finish the response in one paragraph.
        """

    # brevity_guidance = (
    #     "- Respond with no more than 2-3 short sentences using concise, telegraphic wording. Avoid filler or explanatory phrases."
    #     if not long_answer else
    #     """Structure your response: Brief introduction → Key findings → Clinical implications → Conclusion
    #     Provide thorough explanations, covering all key findings and context when relevant."""
    # )


    prompt_text = f"""
    You are an expert biomedical assistant with access to a hybrid knowledge retrieval system consisting of:
    1. RAPTOR RAG (hierarchical internal biomedical documents)
    2. Web-sourced content (explicitly marked as [Web Source])

    Your task:
    - Answer the question using ONLY the provided context.
    - Synthesize information across retrieved sources when applicable.
    - Clearly distinguish internal knowledge from web-sourced information.
    - Do NOT fabricate or infer beyond the context.
    - If information is missing, explicitly state what is missing.
    - Use a professional, clinical tone suitable for medical professionals.
    - Do NOT use numeric citations like [1], [2], etc.

    {output_rules}

    Question:
    {question}

    Context:
    {context_text}

    Respond STRICTLY according to the OUTPUT FORMAT rules above.
    """

    # prompt = f"""You are an expert biomedical assistant with access to a hybrid knowledge retrieval system combining:
    # 1. RAPTOR RAG (hierarchical knowledge base from internal documents)
    # 2. Real-time web search (for recent information and external sources)

    # The provided context may come from:
    # - Multi-level hierarchical summaries (higher-level = abstract, lower-level = detailed)
    # - Web sources (marked with [Web Source:] for recent/external information)
    # - Information has been retrieved based on semantic similarity to your question

    # Your task is to synthesize information from these sources to provide a comprehensive answer.

    # Guidelines:
    # - Base your answer STRICTLY on the provided context (RAPTOR + Web sources)
    # - Synthesize information across different sources and abstraction levels
    # - When citing internal documents, reference naturally (e.g., "According to the ARTSENS manual...")
    # - When citing web sources, mention them clearly (e.g., "According to recent web research..." or "Web sources indicate...")
    # - Do NOT use numeric citations like [1], [2], etc.
    # - Clearly distinguish between internal knowledge base and web-sourced information when relevant
    # - Maintain professional, clinical tone suitable for medical professionals
    # - If context is insufficient, explicitly state what information is missing
    # - Prioritize accuracy over completeness - never fabricate details
    # {brevity_guidance}

    # Question:
    # {question}

    # Context from Hybrid Retrieval (RAPTOR + Web):
    # {context_text}
    # """

    try:
        llm = ChatOpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            api_key=os.getenv("OLLAMA_API_KEY"),
            model="gpt-oss:20b-cloud",
            temperature=0.1,
        )

        llm_start = time.time()
        
        # Invoke the model
        response = llm.invoke([HumanMessage(content=prompt_text)])
        
        llm_wall_time = time.time() - llm_start
        answer = response.content.strip()

        metadata = response.response_metadata
        
        usage_tracker.record_generation(
            prompt_tokens=int(metadata.get("token_usage", {}).get("prompt_tokens", 0)),
            completion_tokens=int(metadata.get("token_usage", {}).get("completion_tokens", 0)),
            # If your cloud provider doesn't provide durations in ns, use 0 
            # or calculate based on wall time
            prompt_duration_ns=0, 
            eval_duration_ns=0,
            total_duration_ns=0,
            wall_time_s=llm_wall_time,
        )
        
        # Clean up the response
        final_answer = f"\n\n✅ Answer\n{'='*70}\n\n" + answer.replace('_', r'\_')
        if show_sources and sources:
            source_list = '\n\n'.join([f"**{i}. {s['file']}**\n{s['content']}" for i, s in enumerate(sources, 1)])
            final_answer += f"\n\n💡 Sources consulted\n{'='*70}\n\n{source_list}"

        return final_answer

    except Exception as e:
        logger.error(f"Cloud LLM error: {str(e)}")
        return f"❌ Error: {str(e)}"
        
    except Exception as e:
        error_msg = f"❌ Error generating answer: {str(e)}\nPlease check if Ollama is running and the model '{CONFIG['model']}' is available."
        logger.error(f"LLM error: {str(e)}")
        return error_msg
    

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
            
            # if query.lower() == 'stats':
            #     try:
            #         with open("query_history.json", "r", encoding="utf-8") as f:
            #             history = json.load(f)
            #         print(f"\n📊 Query Statistics:")
            #         print(f"• Total queries: {len(history)}")
            #         if history:
            #             avg_time = sum(entry.get('response_time', 0) for entry in history) / len(history)
            #             print(f"• Average response time: {avg_time:.2f} seconds")
            #             recent_query = history[-1]
            #             print(f"• Last query: {recent_query['timestamp']}")
            #     except FileNotFoundError:
            #         print("\n📊 No query history found yet.")
            #     continue
            
            if not query:
                print("⚠️  Please enter a valid question.")
                continue
            
            print("🔍 Retrieving relevant information (RAPTOR + Web + Query Refinement)...")
            start_time = time.time()
            
            # Use unified raptor_retrieve with all features enabled
            result = raptor_retrieve(
                query, 
                summary_tree, 
                top_k_root=CONFIG["default_top_k_root"], 
                top_k_children=CONFIG["default_top_k_children"],
                use_query_refinement=False,  # Enable query refinement for better embeddings
                use_web_search=True,  # Enable web search for recent/insufficient results
                max_web_results=3,
                enable_keyword_rescue=False
                
            )
            stop_time = time.time()
            retrieval_time = stop_time - start_time
            print(f"⏱️  Retrieval completed in {retrieval_time:.2f} seconds")

            all_contexts = result['all_contexts']
            raptor_count = result['sources']['raptor']
            web_count = result['sources']['web']
            
            if not all_contexts:
                print("❌ No relevant information found. Try rephrasing your question.")
                continue

            # Limit total contexts if needed
            results = all_contexts[:MAX_CONTEXTS]
            
            print(f"📚 Retrieved {raptor_count} RAPTOR chunks + {web_count} web sources")
            print("\n🤖 Generating comprehensive answer...\n")
            
            # Generate answer
            answer = answer_llm(query, results, show_sources=True, web_results=result['web_results'], long_answer=False)
            response_time = time.time() - start_time
            
            # Extract sources for history
            sources = set()
            for doc in results:
                # Handle both dict and Document objects
                if isinstance(doc, dict):
                    metadata = doc.get('metadata', {})
                    source = metadata.get('file_id', 'Unknown source')
                else:
                    source = doc.metadata.get('file_id', 'Unknown source')
                
                if source != 'Unknown source':
                    clean_source = source.replace('_manual_extracted', ' Manual').replace('_', ' ').title()
                    sources.add(clean_source)
                else:
                    sources.add(source)
            
            # print("=" * 70)
            # print("📋 ANSWER:")
            # print("=" * 70)
            answer_prefix = "Answer\n"+"="*70+"\n"
            markdown_answer = Markdown(answer)
            console.print(markdown_answer)
            print("=" * 70)
            print(f"⏱️  Response time: {response_time:.2f} seconds")
            
            # Save to history
            # save_query_history(query, answer, list(sources), response_time)
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again with a different question.")


def single_query(question: str, top_k_root: Optional[int] = None, top_k_children: Optional[int] = None,
                 show_sources: bool = True, enable_web_search: bool = True,
                 enable_query_refinement: bool = True, enable_keyword_rescue: bool = True,
                 long_answer: bool = False, return_usage: bool = False) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """Process a single query and return the answer using RAPTOR retrieval."""
    top_k_root = top_k_root or CONFIG["default_top_k_root"]
    top_k_children = top_k_children or CONFIG["default_top_k_children"]
    
    logger.info(f"Processing single query: {question[:100]}...")
    usage_tracker.reset()
    overall_start = time.time()

    # Use unified raptor_retrieve
    try:
        start_time = time.time()
        result = raptor_retrieve(
            question, 
            summary_tree, 
            top_k_root=top_k_root, 
            top_k_children=top_k_children,
            use_query_refinement=enable_query_refinement,
            use_web_search=enable_web_search,
            max_web_results=3,
            enable_keyword_rescue=enable_keyword_rescue
        )
        stop_time = time.time()
        retrieval_time = stop_time - start_time
        print(f"Retrieval time: {retrieval_time:.2f} seconds")

        results = result['all_contexts'][:MAX_CONTEXTS]
        answer = answer_llm(
            question,
            results,
            show_sources=show_sources,
            web_results=result['web_results'],
            long_answer=long_answer
        )
    finally:
        usage_tracker.finalize(time.time() - overall_start)
    
    # Extract sources for history
    sources = set()
    for doc in results:
        # Handle both dict and Document objects
        if isinstance(doc, dict):
            metadata = doc.get('metadata', {})
            source = metadata.get('file_id', 'Unknown source')
        else:
            source = doc.metadata.get('file_id', 'Unknown source')
        
        if source != 'Unknown source':
            clean_source = source.replace('_manual_extracted', ' Manual').replace('_', ' ').title()
            sources.add(clean_source)
        else:
            sources.add(source)
    
    # Save to history
    # save_query_history(question, answer, list(sources), usage_tracker.snapshot().get("total_wall_time_s", 0.0))
    
    if return_usage:
        return answer, usage_tracker.snapshot()

    return answer


if __name__ == "__main__":
    # Run interactive mode by default
    interactive_query()