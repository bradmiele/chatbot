from collections import defaultdict
import pickle
import datetime
import faiss
import os
import numpy as np
from utils import get_embedding

class VectorStore:
    def __init__(self, dimension, index_path=None, mapping_path=None):
        self.dimension = dimension
        self.index_path = index_path or "faiss_index.index"
        self.mapping_path = mapping_path or "id_to_text.pkl"
        self.index = faiss.IndexFlatL2(dimension)
        self.id_to_text = defaultdict(list)
        self.next_id = 0

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, 'rb') as f:
                self.id_to_text = pickle.load(f)
                if self.id_to_text:
                    self.next_id = max(max(v.keys()) for v in self.id_to_text.values()) + 1

    def add(self, embedding, entry):
        embedding = np.array(embedding).reshape(1, -1)  # Convert to NumPy array before adding
        self.index.add(embedding)
    
        entry_id = self.next_id
        self.next_id += 1
    
        session_id = entry['session_id']
        if session_id not in self.id_to_text:
            self.id_to_text[session_id] = {}

        self.id_to_text[session_id][entry_id] = entry

    def search(self, query_embedding, k=5):
        query_embedding = np.array(query_embedding).reshape(1, -1)  # Convert to NumPy array and reshape
        distances, indices = self.index.search(query_embedding, k)
    
        results = []
        for idx in indices[0]:
            for session_entries in self.id_to_text.values():
                if idx in session_entries:
                    results.append(session_entries[idx])
        return results

    def retrieve_recent(self, session_id, limit=5):
        if session_id not in self.id_to_text:
            return []
        
        entries = list(self.id_to_text[session_id].values())
        sorted_entries = sorted(entries, key=lambda x: x['timestamp'], reverse=True)
        
        return sorted_entries[:limit]

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.mapping_path, 'wb') as f:
            pickle.dump(self.id_to_text, f)
    
    def chunk_session_transcript(self, session_log):
        """
        Accepts a list of log entries and breaks them into summary-friendly chunks.
        """
        chunks = []
        current_chunk = ""

        for entry in session_log:
            role = entry['type']
            content = entry['content'].strip()
            timestamp = entry['timestamp']
            formatted = f"[{role.capitalize()} @ {timestamp}]: {content}\n"

            if len(current_chunk + formatted) > 1000:  # arbitrary chunk size
                chunks.append(current_chunk.strip())
                current_chunk = formatted
            else:
                current_chunk += formatted

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def store_transcript(self, session_log, session_id):
        chunks = self.chunk_session_transcript(session_log)

        for chunk in chunks:
            entry = {
                "content": chunk,
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": session_id,
                "type": "transcript"
            }
            embedding = get_embedding(chunk)
            self.add(embedding, entry)
        
        self.save()

