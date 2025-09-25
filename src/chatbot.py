import faiss
import json
from sentence_transformers import SentenceTransformer

# Load embeddings and responses
index = faiss.read_index("../models/faiss_index.bin")
with open("../models/responses.json", "r", encoding="utf-8") as f:
    responses = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_response(user_input, top_k=1):
    embedding = model.encode([user_input])
    D, I = index.search(embedding, top_k)
    return responses[I[0][0]]

print("🤖 Chatbot is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("bot: Goodbye! Have a great day! ")
        break
    reply = get_response(user_input)
    print(f"bot: {reply}")
