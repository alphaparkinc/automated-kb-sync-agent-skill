import os
import re
from typing import List, Dict, Any, Optional

class KBSyncClient:
    """
    Client SDK for cleaning raw multi-channel documents, extracting structured QA pairs,
    and prepping database chunks for vector ingestion.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("KB_SYNC_API_KEY")
        self.mock_mode = self.api_key is None or self.api_key == "mock"

    def extract_qa_pairs(self, content: str, source_id: str) -> List[Dict[str, Any]]:
        """
        Parses text for explicit or implicit QA patterns, producing clean structured outputs.
        """
        qa_pairs = []
        # Look for custom Q: / A: patterns or lines containing questions
        lines = content.split('\n')
        current_q = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple heuristic matching Q: or Question:
            if re.match(r'^(Q:|Question:|Que:|QUE:)', line, re.IGNORECASE):
                current_q = re.sub(r'^(Q:|Question:|Que:|QUE:)\s*', '', line, flags=re.IGNORECASE)
            elif re.match(r'^(A:|Answer:|:|:)', line, re.IGNORECASE) and current_q:
                ans = re.sub(r'^(A:|Answer:|:|:)\s*', '', line, flags=re.IGNORECASE)
                qa_pairs.append({
                    "question": current_q,
                    "answer": ans,
                    "metadata": {
                        "source_id": source_id,
                        "confidence_score": 0.95
                    }
                })
                current_q = None
            elif line.endswith('?') and len(line) > 10:
                current_q = line

        # If we didn't find formal Q&A patterns, extract paragraphs as chunks
        if not qa_pairs and len(content) > 30:
            qa_pairs.append({
                "question": f"Summarized info from {source_id}",
                "answer": content.strip().replace('\n', ' '),
                "metadata": {
                    "source_id": source_id,
                    "confidence_score": 0.70
                }
            })
            
        return qa_pairs

    def generate_chunks(self, content: str, chunk_size: int, source_id: str) -> List[Dict[str, Any]]:
        """
        Splits content into windowed chunks for text embedding.
        """
        words = content.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size])
            chunks.append({
                "text": chunk_text,
                "vector_meta": {
                    "source_id": source_id,
                    "word_count": len(words[i:i+chunk_size]),
                    "chunk_index": len(chunks)
                }
            })
        return chunks
