import faiss
import json
from sentence_transformers import SentenceTransformer

dataset_file = "../data/dataset.txt"
corpus = []
responses = []

with open(dataset_file, "r", encoding="utf-8") as f:
    for line in f:
        # Skip empty lines
        if not line.strip():
            continue

        # Try splitting by tab or '||' or fallback
        if "\t" in line:
            user_text, bot_text = line.strip().split("\t")
        elif "||" in line:
            user_text, bot_text = line.strip().split("||")
        else:
            # If no separator, skip this line
            print(f"Skipping line (invalid format): {line}")
            continue

        corpus.append(user_text.strip())
        responses.append(bot_text.strip())

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = model.encode(corpus)

# Create FAISS index
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

# Save index and responses
faiss.write_index(index, "../models/faiss_index.bin")
with open("../models/responses.json", "w", encoding="utf-8") as f:
    json.dump(responses, f, indent=2)

print("Embeddings and index created!")
