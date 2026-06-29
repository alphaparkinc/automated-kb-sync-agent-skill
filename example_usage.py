import sys
import json
from kb_sync import KBSyncClient

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("=== Automated KB Sync Agent Example ===")
    client = KBSyncClient()
    
    raw_notion_page = (
        "Question: How do users request a custom refund refund window extension?\n"
        "Answer: Users must open a priority support ticket under billing and select 'dispute extension'.\n"
        "Question: What is the typical validation duration for new B2B accounts?\n"
        "Answer: Verification is generally completed within 2 business days after submission of corporate tax IDs."
    )
    
    print("\n--- Extracting QA Pairs ---")
    qa_pairs = client.extract_qa_pairs(raw_notion_page, "notion-billing-v2")
    print(json.dumps(qa_pairs, indent=2))
    
    print("\n--- Generating Vector Chunks ---")
    chunks = client.generate_chunks(raw_notion_page, chunk_size=20, source_id="notion-billing-v2")
    print(json.dumps(chunks, indent=2))

if __name__ == "__main__":
    main()
