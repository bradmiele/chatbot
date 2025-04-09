import numpy as np  
import json
import re
import os
import requests
import datetime
from vector_store import VectorStore
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

# configureable settings
MEMORY_MODE = "summary"  # or "full"
os.environ["OMP_NUM_THREADS"] = "1" #otherwise shit breaks

def summarize_context(entries):
    context_lines = []

    for msg in entries:
        role = msg.get("type", "unknown")
        timestamp = msg.get("timestamp", "")
        content = msg.get("summary", msg.get("content", "")).strip()

        if role == "user":
            context_lines.append(f"[User @ {timestamp}]: {content}")
        elif role in ("assistant", "chatbot"):
            context_lines.append(f"[Assistant @ {timestamp}]: {content}")
        else:
            context_lines.append(f"[{role} @ {timestamp}]: {content}")

    return "\n".join(context_lines)

def create_log_entry(role, content, session_id):
    """
    Creates a standardized log entry for storing in the vector store.
    Accepts role ('user' or 'assistant'), not legacy 'chatbot'.
    """
    assert role in ("user", "assistant"), f"Invalid role: {role}"

    return {
        "content": content.strip(),
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": session_id,
        "type": role
    }

def format_message(entry):
    """
    Normalize stored entries into OpenAI-compatible message objects.
    Ensures proper role assignment and strips any leftover prefixes.
    """
    role_map = {
        "user": "user",
        "assistant": "assistant",
        "chatbot": "assistant",  # backward compatibility
        "bot": "assistant",      # just in case
    }

    role = role_map.get(entry.get("type", "user"), "user")
    content = entry.get("content", "").strip()

    # Strip legacy prefixes if they exist
    if content.lower().startswith("user:"):
        content = content[5:].strip()
    elif content.lower().startswith("chatbot:") or content.lower().startswith("assistant:"):
        content = content.split(":", 1)[1].strip()

    return {"role": role, "content": content}


def load_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def build_system_prompt(prompt_file="prompt.txt", tone_file="tone.txt", expertise_file="expertise.txt"):
    prompt = load_file(prompt_file)
    tone = load_file(tone_file)
    expertise = load_file(expertise_file)

    combined_prompt = f"""
{prompt}

## Tone Guidance:
{tone}

## Areas of Expertise (phd level):
{expertise}

Use the provided context and guidelines above when responding.
    """.strip()

    return combined_prompt

def get_embedding(text):
    return embedding_model.encode(text)

def get_similar_context(query, store, k=5):
    query_embedding = get_embedding(query)
    results = store.search(query_embedding, k)
    return results  # Return full entries, not just text

def process_response(text):
    """
    Replace em-dashes with a period followed by an uppercase letter.
    """
    def replacer(match):
        # Extract the letter following the em-dash
        following_text = match.group(1)
        if following_text:
            return f". {following_text.upper()}"
        return "."
    
    # Replace em-dashes with a period and capitalize the following character
    processed_text = re.sub(r"—\s*([a-zA-Z])", replacer, text)
    
    # If an em-dash is followed by a non-letter character or nothing, replace it with a period
    processed_text = re.sub(r"—", ".", processed_text)
    
    return processed_text

def generate_local_response(messages, model="mistral"):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

def generate_response(user_message, store, system_prompt, session_id, memory_mode=MEMORY_MODE, history_limit=5):   
    if memory_mode == "summary":
        context_messages = get_similar_context(user_message, store, k=history_limit)
    
        summarized_context = "\n\nRelevant background:\n" + "\n".join([
            msg.get("summary", msg.get("content", "")).strip()
            for msg in context_messages
        ]) if context_messages else ""

        messages = [
            {"role": "system", "content": f"{system_prompt}{summarized_context}"},
            {"role": "user", "content": user_message}
        ]

    elif memory_mode == "full":
        messages = [{"role": "system", "content": system_prompt}]
        recent_history = store.retrieve_recent(session_id, limit=history_limit)

        for entry in reversed(recent_history):
            messages.append(format_message(entry))

        messages.append({"role": "user", "content": user_message})

    # Debug output
    print("Sending the following messages to API:")
    print(json.dumps(messages, indent=2))

    # Generate assistant response
    answer = generate_local_response(messages)
    processed_answer = process_response(answer)
    timestamp = datetime.datetime.now().isoformat()

    # Log entries
    user_entry = create_log_entry("user", user_message, session_id)
    chatbot_entry = create_log_entry("assistant", answer, session_id)

    # Vector store update
    store.add(np.array(get_embedding(user_entry['content'])), user_entry)
    store.add(np.array(get_embedding(chatbot_entry['content'])), chatbot_entry)
    store.save()

    return processed_answer

if __name__ == "__main__":
    dimension = embedding_model.get_sentence_embedding_dimension()
    store = VectorStore(dimension, index_path="faiss_index.index", mapping_path="id_to_text.pkl")
    system_prompt = build_system_prompt()

    session_id = f"session_{datetime.datetime.now().timestamp()}"
    
    print("Chatbot is ready. Type 'exit' or 'quit' to end.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = generate_response(user_input, store, system_prompt, session_id)
        print("Chatbot:", response)
