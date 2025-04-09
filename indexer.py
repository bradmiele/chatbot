# indexer.py
import os
import glob
import tiktoken
from vector_store import VectorStore
from chatbot import get_embedding

# Set up tokenizer for the OpenAI embeddings model
tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, max_tokens=1000):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def index_files(directory, vector_store):
    file_paths = glob.glob(os.path.join(directory, "*.txt"))
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            chunks = chunk_text(content)
            for idx, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)
                text_entry = f"File: {file_path}, Chunk: {idx}\n{chunk}"
                vector_store.add(embedding, text_entry)
                print(f"Indexed {file_path}, chunk {idx}")
    vector_store.save()

if __name__ == "__main__":
    dimension = embedding_model.get_sentence_embedding_dimension()
    store = VectorStore(dimension)
    directory_to_index = "./data"
    index_files(directory_to_index, store)
