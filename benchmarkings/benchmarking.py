import csv
import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
from query_llm import *
from mechanics.ollama_usage_tracker import usage_tracker

load_dotenv()

# -------------------------
# Question list
# -------------------------
with open("question_bank_20.json","r") as f:
    questions = json.loads(f.read())

# -------------------------
# Utility: Extract answer/sources
# -------------------------
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


# -------------------------
# File setup and recovery
# -------------------------
output_path = "Timing Precise 20 Question Benchmarking - Saransh.txt"
usage_report_path = "benchmark_usage_report.csv"
USAGE_COLUMNS = [
    "question_number",
    "question",
    "embedding_calls",
    "embedding_prompt_tokens",
    "embedding_eval_tokens",
    "embedding_wall_time_s",
    "generation_prompt_tokens",
    "generation_completion_tokens",
    "generation_wall_time_s",
    "total_wall_time_s",
]


def _init_usage_report(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(USAGE_COLUMNS)


def _append_usage_row(question_number: int, question_text: str, metrics: dict) -> None:
    embedding = metrics.get("embedding", {}) if metrics else {}
    generation = metrics.get("generation", {}) if metrics else {}
    row = [
        question_number,
        question_text,
        embedding.get("calls", 0),
        embedding.get("prompt_tokens", 0),
        embedding.get("eval_tokens", 0),
        round(embedding.get("wall_time_s", 0.0), 4),
        generation.get("prompt_tokens", 0),
        generation.get("completion_tokens", 0),
        round(generation.get("wall_time_s", 0.0), 4),
        round(metrics.get("total_wall_time_s", 0.0) if metrics else 0.0, 4),
    ]
    with open(usage_report_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

answers, sources = [], []
_init_usage_report(usage_report_path)

# Recover progress if file exists
if os.path.exists(output_path):
    print(f"📄 Resuming from existing file: {output_path}")
    with open(output_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Skip header line if present
    processed_lines = [line.strip() for line in lines[1:] if line.strip()]
    for line in processed_lines:
        # Split safely using '$$$' separator, not breaking on commas
        parts = line.split(" $$$ ", 2)
        if len(parts) == 3:
            _, answer, source = parts
            answers.append(answer.replace("\\n", "\n"))
            sources.append(source.replace("\\n", "\n"))
        else:
            answers.append("")
            sources.append("")
else:
    print("🆕 Starting fresh benchmarking session.")
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("Question $$$ Original Response $$$ RAG Response $$$ RAG Sources $$$ Original Document\n")

# Determine where to resume
start_index = len(answers)
total_questions = len(questions)

print(f"🔁 Resuming from question {start_index+1} of {total_questions} ({start_index} already done).")

# -------------------------
# Main loop
# -------------------------
with open(output_path, "a", encoding="utf-8-sig") as f:
    for index in range(start_index, total_questions):
        question_idx = questions[index]
        question = question_idx["question"]
        original_answer = question_idx["answer"]
        original_file = question_idx["source_document"]
        print(f"\n🔍 Processing question {index+1}/{total_questions}...")

        usage_metrics = {}
        try:
            rag_response, usage_metrics = single_query(
                question, 
                top_k_root=10, 
                top_k_children=2, 
                show_sources=True, 
                enable_web_search=True, 
                enable_query_refinement=False,
                enable_keyword_rescue=False,
                long_answer=False,
                return_usage=True
            )
            if not rag_response:
                print("⚠️ Empty response.")
                answers.append("")
                sources.append("")
                continue

            answer_text, sources_text = extract_answer_sources(rag_response)

            safe_answer = answer_text.replace("\n", "\\n")
            safe_sources = sources_text.replace("\n", "\\n")

            # Append to file
            f.write(f"{question} $$$ {original_answer} $$$ {safe_answer} $$$ {safe_sources} $$$ {original_file}\n")
            f.flush()

            answers.append(answer_text)
            sources.append(sources_text)
            embedding_tokens = usage_metrics.get("embedding", {}).get("prompt_tokens", 0)
            generation_prompt = usage_metrics.get("generation", {}).get("prompt_tokens", 0)
            generation_completion = usage_metrics.get("generation", {}).get("completion_tokens", 0)
            total_time = usage_metrics.get("total_wall_time_s", 0.0)
            print(
                "✅ Success: Question {} | 📊 Embedding tokens: {} | Generation tokens (prompt/completion): {}/{} | ⏱️ {:.2f}s".format(
                    index + 1,
                    embedding_tokens,
                    generation_prompt,
                    generation_completion,
                    total_time,
                )
            )

        except Exception as e:
            print(f"❌ Error for question {index+1}: {e}")
            answers.append("")
            sources.append("")
            usage_metrics = usage_tracker.snapshot()
        finally:
            _append_usage_row(index + 1, question, usage_metrics)

