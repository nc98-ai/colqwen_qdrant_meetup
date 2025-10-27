#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qa_from_qdrant_with_ollama_fixed2.py

Script QA utilisant :
 - Qdrant (named vectors)
 - embedder ColPali ou CLIP pour encoder la question
 - Ollama local via CLI pour la génération

Corrige le format de la requête de recherche pour les named vectors
(attendu par qdrant-client v1.14.x).
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from typing import List, Dict, Any

import numpy as np
import torch
from qdrant_client import QdrantClient

# Embedders
from transformers import CLIPProcessor, CLIPModel
try:
    from transformers import ColPaliProcessor, ColPaliForRetrieval
    HAVE_COLPALI = True
except Exception:
    HAVE_COLPALI = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ClipTextEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Init CLIP text embedder on device %s (model=%s)", self.device, model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_text(self, text: str) -> List[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        vec = feats.cpu().numpy()[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(float).tolist()


class ColPaliTextEmbedder:
    def __init__(self, model_name="vidore/colpali-v1.3-hf", device=None):
        if not HAVE_COLPALI:
            raise RuntimeError("ColPali not available in this installation. Install proper transformers version or use --embedder clip.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Init ColPali embedder on device %s (model=%s)", self.device, model_name)
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        self.model = ColPaliForRetrieval.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _extract_embedding_from_output(self, out) -> np.ndarray:
        # try common attribute names
        for name in ("embeddings", "image_embeds", "text_embeds", "image_embeds_0", "image_embedding", "last_hidden_state"):
            if hasattr(out, name):
                val = getattr(out, name)
                if isinstance(val, torch.Tensor):
                    return val.cpu().numpy()
                try:
                    arr = np.asarray(val)
                    if arr.size:
                        return arr
                except Exception:
                    pass
        if isinstance(out, (tuple, list)) and len(out) > 0:
            cand = out[0]
            if isinstance(cand, torch.Tensor):
                return cand.cpu().numpy()
            try:
                arr = np.asarray(cand)
                if arr.size:
                    return arr
            except Exception:
                pass
        raise RuntimeError("Cannot extract embedding tensor from ColPali output.")

    def embed_text(self, text: str) -> List[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        emb = self._extract_embedding_from_output(out)
        emb = np.asarray(emb)
        # common shapes -> reduce to 1-D
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)
        if emb.ndim == 2:
            emb = emb.mean(axis=0)
        emb = emb.reshape(-1)
        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.tolist()


def retrieve_top_k(client: QdrantClient, collection_name: str, vector: List[float], k: int = 5,
                   vector_name: str = None, with_payload: bool = True):
    """
    Construct and call client.search with correct named-vector format for this qdrant-client version.
    For named vectors we build: {"name": vector_name, "vector": vector}
    """
    if vector_name:
        query_vector = {"name": vector_name, "vector": vector}
    else:
        query_vector = vector

    # call search (avoid passing unknown kwargs that older/newer clients may reject)
    try:
        results = client.search(collection_name=collection_name,
                                query_vector=query_vector,
                                limit=k,
                                with_payload=with_payload)
    except AssertionError as ae:
        logging.warning("client.search assertion error, retrying without additional kwargs: %s", ae)
        results = client.search(collection_name=collection_name,
                                query_vector=query_vector,
                                limit=k)
    return results


def call_ollama_cli(model: str, prompt: str, timeout: int = 60, ollama_bin: str = "ollama") -> str:
    cmd = [ollama_bin, "run", model, "--prompt", prompt]
    logging.info("Calling Ollama CLI: %s %s ...", ollama_bin, model)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            logging.error("Ollama CLI failed (rc=%d). stderr: %s", proc.returncode, proc.stderr.strip())
            raise RuntimeError(f"Ollama CLI failed: {proc.stderr.strip()}")
        return proc.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("ollama binary not found on PATH. Install Ollama or adjust --ollama-cli.")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Ollama run timed out after {timeout}s")


def build_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    prompt_parts = []
    prompt_parts.append("Tu es un assistant utile. Réponds à la question suivante en te basant uniquement sur les extraits fournis.\n")
    prompt_parts.append("Question: " + question + "\n\n")
    prompt_parts.append("Extraits récupérés (numérotés) :\n")
    for i, r in enumerate(retrieved, start=1):
        # r can be PointStruct-like or dict
        payload = getattr(r, "payload", None) or (r.get("payload") if isinstance(r, dict) else {})
        score = getattr(r, "score", None) or (r.get("score") if isinstance(r, dict) else None)
        pdf_fn = payload.get("pdf_filename") if isinstance(payload, dict) else None
        page = payload.get("page_number") if isinstance(payload, dict) else None
        text = payload.get("text") if isinstance(payload, dict) else None
        snippet = (text or "").replace("\n", " ").strip()
        if len(snippet) > 800:
            snippet = snippet[:800].rsplit(" ", 1)[0] + "…"
        prompt_parts.append(f"[{i}] file: {pdf_fn} | page: {page} | score: {score}\n{snippet}\n")
    prompt_parts.append("\nRéponds en français de façon concise. Si l'information n'est pas présente dans les extraits, dis 'Je ne sais pas' et ne fais pas d'invention.\n")
    return "\n".join(prompt_parts)


def main():
    parser = argparse.ArgumentParser(description="QA from Qdrant using ColPali/CLIP embeddings + Ollama local")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embedder", choices=("colpali", "clip"), default="colpali")
    parser.add_argument("--model", default="vidore/colpali-v1.3-hf", help="Name for embedder model")
    parser.add_argument("--vector-name", default="page_image", help="Named vector key in Qdrant")
    parser.add_argument("--ollama-model", default="gemma3:4b", help="Ollama local model")
    parser.add_argument("--ollama-cli", default="ollama", help="Path to ollama binary")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant, api_key=args.api_key)
    logging.info("Connected to Qdrant %s (collection=%s)", args.qdrant, args.collection)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Using device: %s", device)
    if args.embedder == "colpali":
        if not HAVE_COLPALI:
            logging.error("ColPali not available. Use --embedder clip")
            sys.exit(1)
        embedder = ColPaliTextEmbedder(model_name=args.model, device=device)
    else:
        embedder = ClipTextEmbedder(model_name=args.model, device=device)

    # embed query
    t0 = time.time()
    logging.info("Embedding query using %s embedder...", args.embedder)
    query_vec = embedder.embed_text(args.query)
    logging.info("Query embedded (dim=%d) in %.3fs", len(query_vec), time.time() - t0)

    # retrieve
    logging.info("Query Qdrant: collection=%s k=%d vector_name=%s", args.collection, args.k, args.vector_name)
    try:
        results = retrieve_top_k(client, args.collection, vector=query_vec, k=args.k, vector_name=args.vector_name, with_payload=True)
    except Exception as e:
        logging.exception("Search failed: %s", e)
        sys.exit(1)

    logging.info("Retrieved %d results", len(results))

    prompt = build_prompt(args.query, results)

    try:
        t2 = time.time()
        reply = call_ollama_cli(args.ollama_model, prompt, timeout=args.timeout, ollama_bin=args.ollama_cli)
        logging.info("Ollama finished in %.2fs", time.time() - t2)
    except Exception as e:
        logging.exception("Ollama call failed: %s", e)
        sys.exit(1)

    print("\n\n=== PROMPT SENT TO OLLAMA (truncated) ===\n")
    print(prompt[:5000])
    print("\n\n=== OLLAMA RESPONSE ===\n")
    print(reply)
    print("\n\n=== RETRIEVED ITEMS (brief) ===\n")
    for i, r in enumerate(results, start=1):
        payload = getattr(r, "payload", None) or (r.get("payload") if isinstance(r, dict) else {})
        pdf_fn = payload.get("pdf_filename") if isinstance(payload, dict) else None
        page = payload.get("page_number") if isinstance(payload, dict) else None
        score = getattr(r, "score", None) or (r.get("score") if isinstance(r, dict) else None)
        snippet = (payload.get("text")[:300].replace("\n", " ") if isinstance(payload, dict) and payload.get("text") else "")
        print(f"[{i}] {pdf_fn} p{page} score={score} -> {snippet}")


if __name__ == "__main__":
    main()
