# scripts/build_index.py
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from ragdn.config import CHUNKS_PATH, EMB_PATH, INDEX_DIR, META_PATH, MODEL_EMBEDDING
from ragdn.index import chunks_sha256

with CHUNKS_PATH.open(encoding="utf-8") as f:
    chunks = [json.loads(l) for l in f]

embedder = SentenceTransformer(MODEL_EMBEDDING)
emb = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True,
                      batch_size=16, show_progress_bar=True).astype(np.float32)

INDEX_DIR.mkdir(exist_ok=True)
np.save(EMB_PATH, emb)
META_PATH.write_text(json.dumps({
    "model": MODEL_EMBEDDING, "n": len(chunks), "dim": int(emb.shape[1]),
    "content_sha256": chunks_sha256(chunks),
}, indent=2), encoding="utf-8")
