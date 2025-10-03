from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)

def create_chunks(corpus, file_id):
    chunks, indexer = [], 0
    splits = text_splitter.split_text(corpus["text"])
    for split in splits:
        chunks.append({
            "id": f"file_{file_id}_chunk_{indexer}_level_0",
            "text": split,
            "source": corpus["source"]
        })
        indexer += 1
    return chunks


with open("./essentials/group_chunks.json", "r", encoding="utf-16") as f:
    doc_corpus = json.loads(f.read())
    chunk_generator = {}
    for file_id, file_content in enumerate(doc_corpus):
        chunks = create_chunks(file_content, file_id)
        chunk_generator[f"file_{file_id}"] = {"levels": {"level_0": chunks}, "source": file_content["source"]}

    with open("./essentials/summary_tree.json", "w", encoding="utf-16") as out_f:
        json.dump(chunk_generator, out_f, ensure_ascii=False, indent=4)