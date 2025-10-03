import os
import json


all_chunks = []
for i in os.listdir("./data/extracted_pdfs"):
    with open(os.path.join("./data/extracted_pdfs", i), "r", encoding="utf-16") as f:
        corpus = json.load(f)
        all_text = ' '.join([j['text'] for j in corpus])
        all_chunks.append({"text": all_text, "source": i.replace(".json",".pdf")})

os.makedirs("./essentials", exist_ok=True)
with open("./essentials/group_chunks.json", "w", encoding="utf-16") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=4)