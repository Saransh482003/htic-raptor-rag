from query_llm import *
import re

question = "Can we wash the polar strap at 45 degrees Celsius after bleaching it?"

def extract_answer_sources(rag_response: str):
    """Extracts the cleaned answer and sources from a RAG response."""
    answer_match = re.search(r"✅ Answer[\s\S]*?(?=💡 Sources consulted|$)", rag_response)
    answer_text = ""
    if answer_match:
        answer_text = re.sub(r"(✅ Answer|=+)", "", answer_match.group(0)).strip()

    sources_match = re.search(r"💡 Sources consulted[\s\S]*", rag_response)
    sources_text = ""
    if sources_match:
        sources_text = re.sub(r"(💡 Sources consulted|=+)", "", sources_match.group(0)).strip()

    return answer_text, sources_text


rag_response = single_query(question, 5, 2, True)
answer_text, sources_text = extract_answer_sources(rag_response)

print(answer_text)
print(sources_text)