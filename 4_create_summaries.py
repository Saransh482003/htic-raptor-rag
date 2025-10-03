import json
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import warnings
warnings.filterwarnings("ignore")


def create_summaries(batch, level):
    llm = ChatOllama(model="llama3", temperature=0)

    summary_prompt = PromptTemplate(
        input_variables=["text", "level"],
        template="""
            You are an expert scientific summarizer for a retrieval-augmented generation (RAG) system.  
            Your task is to create a **concise but information-rich summary** of the following text.  
            The summary will later be used recursively to build a hierarchical knowledge tree (RAPTOR), so it must be coherent, self-contained, and faithful.

            ### Guidelines:
            - **Faithfulness**: Do NOT introduce facts not present in the input text. Summarize only what is given.  
            - **Coverage**: Capture the most important entities, concepts, and relationships mentioned in the text.  
            - **Abstraction**: Shorten long explanations while preserving key details (e.g., thresholds, conditions, findings).  
            - **Clarity**: Write in clear, concise prose; avoid repetition, filler, or references.  
            - **Specificity**: Retain critical domain-specific terms (e.g., "electrocardiogram-based analysis", "≥ 5 events per hour").  
            - **Context Independence**: The summary should stand on its own, without requiring the reader to see the original text.  
            - **Length Control**: 
            - For lower-level chunks (level 0-1), produce ~3-5 sentences.  
            - For higher-level summaries (level ≥ 2), focus more on abstraction and generalization, keeping it 2-3 sentences.  

            ### Output format:
            Return only valid JSON with a single key "summary". Example:
            {{
                "summary": "Concise summary text here."
            }}

            ### Input text (level {level}):
            <<<{text}>>>
        """
    )

    text = "\n\n".join([chunk["text"] for chunk in batch])
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    response = summary_chain.run(text=text, level=level)
    summary = json.loads(response[response.find('{'):response.rfind('}')+1])
    return summary


with open('essentials/summary_tree.json', 'r', encoding="utf-16") as f:
    summary_tree = json.load(f)

for file in summary_tree.keys():
    print(f"Processing {file}...")

    summary_levels = summary_tree[file]["levels"]
    max_level = list(summary_levels.keys())[-1]

    while len(summary_levels[max_level]) > 1:
        print("Current Level:", max_level)
        print("Number of chunks at this level:", len(summary_levels[max_level]))

        current_level = summary_levels[max_level]
        next_level = []
        next_level_num = int(max_level.split('_')[1]) + 1
        next_name = f"level_{next_level_num}"
        batch_size = 5

        print(f"\t🟠 Starting Level: {next_name}...")

        for i in range(0, len(current_level), batch_size):
            batch = current_level[i:i + batch_size]
            summary = create_summaries(batch, next_level_num)
            next_level.append({
                "id": f"{file}_summary_{i//5}_level_{next_level_num}",
                "text": summary["summary"],
                "source": [chunk["id"] for chunk in batch]
            })

            print(f"\t\t🔹Summarized batch {i}-{i+4} -> Summary ID: {file}_summary_{i//5}_level_{next_level_num}")

        summary_levels[next_name] = next_level
        summary_tree[file]["levels"] = summary_levels
        max_level = next_name

        print(f"\t🟢 Completed Level {next_level_num}; Total Chunks: {len(next_level)}")

    print(f"✅ Finished processing {file}.\n")

    with open('essentials/summary_tree.json', 'w', encoding="utf-16") as out_f:
        json.dump(summary_tree, out_f, ensure_ascii=False, indent=4)