import os
import json
from dotenv import load_dotenv

load_dotenv()

all_chunks = []
for i in os.listdir(os.getenv("EXTRACTED_DATA_PATH")):
    with open(os.path.join(os.getenv("EXTRACTED_DATA_PATH"), i), "r", encoding="utf-16") as f:
        corpus = json.load(f)
        all_text = ' '.join([j['text'] for j in corpus])
        all_chunks.append({"text": all_text, "source": i.replace(".json",".pdf")})

os.makedirs(os.getenv("ESSENTIALS_PATH"), exist_ok=True)
with open(os.path.join(os.getenv("ESSENTIALS_PATH"), "group_chunks.json"), "w", encoding="utf-16") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=4)