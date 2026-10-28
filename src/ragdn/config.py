# src/ragdn/config.py
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = ROOT_DIR / "index"

MANIFEST_PATH = DATA_DIR / "manifest.yaml"
GLOSSARY_PATH = DATA_DIR / "glossary.yaml"
ALIASES_PATH = DATA_DIR / "aliases.yaml"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
EMB_PATH = INDEX_DIR / "emb.npy"
META_PATH = INDEX_DIR / "meta.json"

MODEL_EMBEDDING = "BAAI/bge-m3"
MODEL_RERANKER = "BAAI/bge-reranker-v2-m3"
