import json
from pathlib import Path

chunks_path = Path("data/processed/nsw_chunks.jsonl")
chunks = [json.loads(line) for line in open(chunks_path)]

for num in ["41", "42", "84", "110", "159", "162", "170"]:
    matches = [c for c in chunks if c["section_number"] == num and not c.get("schedule")]
    for m in matches:
        print(f"s{num:<5} {m['section_title']}")


for num in ["170", "171", "186", "187", "188"]:
    matches = [c for c in chunks if c["section_number"] == num and not c.get("schedule")]
    for m in matches:
        print(f"s{num:<5} {m['section_title']}")