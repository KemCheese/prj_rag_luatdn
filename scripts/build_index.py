# scripts/build_index.py
import hashlib
import json
import pathlib

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-m3"
with open("data/processed/chunks.jsonl", encoding="utf-8") as f:
    chunks = [json.loads(l) for l in f]
ids = [c["chunk_id"] for c in chunks]

embedder = SentenceTransformer(MODEL)
emb = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True,
                      batch_size=16, show_progress_bar=True).astype(np.float32)

out = pathlib.Path("index"); out.mkdir(exist_ok=True)
np.save(out / "emb.npy", emb)
(out / "meta.json").write_text(json.dumps({
    "model": MODEL, "n": len(ids), "dim": int(emb.shape[1]),
    "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
}, indent=2))