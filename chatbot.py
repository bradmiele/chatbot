import numpy as np  
import json
import re
import os
import requests
import datetime
from vector_store import VectorStore
from openai_integration import generate_openai_response
from utils import get_embedding
from utils import fetch_webpage_text
from utils import model


# configureable settings
MEMORY_MODE = "summary"  # or "full"
os.environ["OMP_NUM_THREADS"] = "1" #otherwise shit breaks
request_counter = 0
PROMPT_REFRESH_INTERVAL = 3  # adjust as needed


def print_debug_messages(messages):
    print("\n=== Sending to LLM ===")
    for m in messages:
        print(f"\n[{m['role'].upper()}]")
        print(m['content'].strip())
    print("======================\n")

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
    assert role in ("user", "assistant"), f"Invalid role: {role}"
    return {
        "content": content.strip(),
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": session_id,
        "type": role
    }

def format_message(entry):
    role_map = {
        "user": "user",
        "assistant": "assistant",
        "chatbot": "assistant",  
        "bot": "assistant",      
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

Use the provided context, tone guidance, and the appropriate areas of expertise above when responding.
    """.strip()

    return combined_prompt

def get_similar_context(query, store, k=5):
    query_embedding = get_embedding(query)
    results = store.search(query_embedding, k)
    return results  # Return full entries, not just text

def get_similar_recent_context(query, store, session_id, search_limit=20, k=5):
    """
    Search recent history only, then select the most semantically relevant.
    """
    recent_entries = store.retrieve_recent(session_id, limit=search_limit)
    if not recent_entries:
        return []

    query_embedding = get_embedding(query)
    scored = []

    for entry in recent_entries:
        entry_embedding = get_embedding(entry['content'])
        score = np.dot(query_embedding, entry_embedding)  # cosine similarity approx
        scored.append((score, entry))

    top_k = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
    return [entry for _, entry in top_k]

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

def generate_local_response(messages, model="mistral:7b-instruct"):
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

def generate_response(user_message, store, system_prompt, session_id, memory_mode=MEMORY_MODE, history_limit=3):   
    if memory_mode == "summary":
        context_messages = get_similar_recent_context(user_message, store, session_id, search_limit=20, k=5)
        context_messages = sorted(context_messages, key=lambda x: x.get("timestamp", ""))
        long_term_hits = []
        
        # Check if it's a long-term recall query
        if re.search(r"do you remember|have we discussed|have I mentioned", user_message, re.IGNORECASE):
            query_embedding = get_embedding(user_message)
            long_term_hits = transcript_store.search(query_embedding, history_limit)

        summarized_context = "\n\n"
        if long_term_hits:
            summarized_context += "Relevant prior discussion:\n" + summarize_context(long_term_hits) + "\n\n"
        elif context_messages:
            summarized_context += "Current context:\n" + summarize_context(context_messages) + "\n\n"

        messages = [
            {"role": "system", "content": f"{system_prompt}{summarized_context.strip()}"},
            {"role": "user", "content": user_message}
        ]

    elif memory_mode == "full":
        messages = [{"role": "system", "content": system_prompt}]
        recent_history = store.retrieve_recent(session_id, limit=5)

        for entry in reversed(recent_history):
            messages.append(format_message(entry))

        messages.append({"role": "user", "content": user_message})

    # Debug output
    print("Sending the following messages to API:")
    print_debug_messages(messages)

    # Generate assistant response
    answer = generate_openai_response(messages)
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
    dimension = model.get_sentence_embedding_dimension()
    store = VectorStore(dimension, index_path="faiss_index.index", mapping_path="id_to_text.pkl")
    transcript_store = VectorStore(dimension, index_path="transcript_index.index", mapping_path="transcript_id_to_text.pkl")
    system_prompt = build_system_prompt()
    session_id = f"session_{datetime.datetime.now().timestamp()}"
    session_log = []       # Holds full conversation history  
    
    print("Chatbot is ready. Type 'exit' or 'quit' to end.")
    while True:
        user_input = input("You: ")
        # Decide which system prompt to send

        if user_input.lower() in ["exit", "quit"]:
            break

        response = generate_response(user_input, store, system_prompt, session_id)

        print("Chatbot:", response)

        # Track the exchange
        user_entry = create_log_entry("user", user_input, session_id)
        assistant_entry = create_log_entry("assistant", response, session_id)
        session_log.extend([user_entry, assistant_entry])
        
        # Transcript Write
        transcript_store.store_transcript(session_log, session_id)
    
  

