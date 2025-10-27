#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script_pdf_to_qdrant_with_colpali_incremental.py (corrigé pour delete points_selector)

Indexe un corpus PDF (./pdf_dir) avec ColPali -> embeddings -> Qdrant, manifest incrémental.
"""

import argparse
import json
import logging
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, Optional, Any

import fitz  # PyMuPDF
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Transformers ColPali (assurez-vous que ces classes existent dans votre env)
from transformers import ColPaliProcessor, ColPaliForRetrieval

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, PointIdsList

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# ---------- config ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
DEFAULT_MODEL = "vidore/colpali-v1.3-hf"
VECTOR_NAME = "page_image"
DISTANCE = Distance.COSINE
RENDER_DPI = 150

# ---------- helpers : manifest ----------
def load_manifest(manifest_path: Path) -> Dict:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            logging.exception("Impossible de lire manifest, démarrage avec manifest vide.")
            return {}
    return {}

def save_manifest(manifest_path: Path, manifest: Dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

def file_signature(path: Path) -> Dict:
    stat = path.lstat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}

# ---------- rendering ----------
def render_page_to_pil(page, dpi=RENDER_DPI):
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# ---------- ColPali embedder ----------
class ColPaliEmbedder:
    def __init__(self, model_name=DEFAULT_MODEL, device: Optional[str]=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Init ColPali (%s) on device %s", model_name, self.device)
        t0 = time.perf_counter()
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        self.model = ColPaliForRetrieval.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._vector_dim = None
        logging.info("Model loaded in %.3fs", time.perf_counter()-t0)

    def _extract_embedding_from_output(self, out):
        if out is None:
            raise RuntimeError("Empty model output.")
        for attr in ("embeddings", "image_embeds", "image_embedding", "image_embeds_0"):
            if hasattr(out, attr):
                val = getattr(out, attr)
                if val is not None:
                    return val
        if isinstance(out, (tuple, list)) and len(out) > 0:
            return out[0]
        return out

    def embed_image(self, pil_image: Image.Image):
        inputs = self.processor(images=pil_image, return_tensors="pt")
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        emb_candidate = self._extract_embedding_from_output(out)
        if isinstance(emb_candidate, torch.Tensor):
            emb = emb_candidate.cpu().numpy()
        else:
            emb = np.asarray(emb_candidate)
        emb = np.asarray(emb)
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(axis=0)
        if emb.ndim == 2:
            emb = emb.mean(axis=0)
        elif emb.ndim > 2:
            emb = emb.reshape(-1)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        if self._vector_dim is None:
            self._vector_dim = int(emb.shape[0])
        if self._vector_dim < 2:
            raise RuntimeError(f"Embedding dimension anormale détectée: {self._vector_dim}")
        return emb.tolist()

    def vector_dim(self):
        return self._vector_dim

# ---------- Qdrant helpers (robuste) ----------
def extract_vectors_config_from_info(info: Any) -> Optional[Dict]:
    if isinstance(info, dict):
        vc = info.get("vectors_config") or info.get("vectors") or info.get("config", {}).get("params", {}).get("vectors")
        if vc:
            return vc
        res = info.get("result")
        if isinstance(res, dict):
            vc = res.get("vectors_config") or (res.get("config") or {}).get("params", {}).get("vectors")
            if vc:
                return vc
    try:
        if hasattr(info, "vectors_config"):
            vc = getattr(info, "vectors_config")
            if vc:
                return vc
    except Exception:
        pass
    try:
        cfg = getattr(info, "config", None)
        if cfg is not None:
            params = getattr(cfg, "params", None)
            if params is not None:
                vecs = getattr(params, "vectors", None)
                if vecs:
                    return vecs
            if isinstance(cfg, dict):
                vecs = (cfg.get("params") or {}).get("vectors")
                if vecs:
                    return vecs
    except Exception:
        pass
    try:
        res = getattr(info, "result", None)
        if res is not None:
            cfg = getattr(res, "config", None)
            if cfg:
                params = getattr(cfg, "params", None)
                if params:
                    vecs = getattr(params, "vectors", None)
                    if vecs:
                        return vecs
            if isinstance(res, dict):
                vecs = (res.get("config") or {}).get("params", {}).get("vectors")
                if vecs:
                    return vecs
    except Exception:
        pass
    try:
        if hasattr(info, "dict"):
            d = info.dict()
            return extract_vectors_config_from_info(d)
    except Exception:
        pass
    return None

def ensure_collection_with_dim(client: QdrantClient, collection_name: str, vector_name: str,
                               vector_dim: int, distance=DISTANCE, force_recreate: bool=False):
    try:
        exists = client.collection_exists(collection_name=collection_name)
    except Exception as e:
        raise RuntimeError(f"Erreur connexion Qdrant: {e}")

    if not exists:
        client.create_collection(collection_name=collection_name,
                                 vectors_config={vector_name: VectorParams(size=vector_dim, distance=distance)})
        logging.info("Collection '%s' créée (vector %s dim=%d).", collection_name, vector_name, vector_dim)
        return

    info = None
    try:
        info = client.get_collection(collection_name=collection_name)
    except Exception as e:
        logging.debug("get_collection error: %s - trying get_collections()", e)
        try:
            allcols = client.get_collections()
            for c in getattr(allcols, "collections", []):
                nm = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)
                if nm == collection_name:
                    info = c
                    break
        except Exception as e2:
            logging.debug("get_collections also failed: %s", e2)

    existing_vectors = None
    if info is not None:
        existing_vectors = extract_vectors_config_from_info(info)

    if not existing_vectors:
        try:
            allcols = client.get_collections()
            for c in getattr(allcols, "collections", []):
                nm = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)
                if nm == collection_name:
                    existing_vectors = extract_vectors_config_from_info(c)
                    if existing_vectors:
                        break
        except Exception:
            pass

    if not existing_vectors:
        logging.error("Impossible de lire vectors_config pour la collection existante. Réponse brute: %r", info)
        raise RuntimeError("Impossible de lire vectors_config pour la collection existante. La récupération a échoué (vérifiez la version du client qdrant et l'API).")

    existing_dim = None
    if isinstance(existing_vectors, dict):
        for key, val in existing_vectors.items():
            if hasattr(val, "size"):
                existing_dim = int(getattr(val, "size"))
                break
            if isinstance(val, dict) and "size" in val:
                existing_dim = int(val["size"])
                break
            if isinstance(val, int):
                existing_dim = int(val)
                break
        if existing_dim is None:
            first_key = next(iter(existing_vectors.keys()))
            fv = existing_vectors[first_key]
            try:
                existing_dim = int(fv)
            except Exception:
                logging.error("Impossible d'extraire la taille du vecteur depuis: %r", fv)
                raise RuntimeError("Impossible d'interpréter vectors_config de la collection existante.")
    else:
        logging.error("vectors_config n'est pas un dict: %s", type(existing_vectors))
        raise RuntimeError("Impossible d'interpréter vectors_config de la collection existante.")

    if existing_dim != vector_dim:
        msg = f"Dimension mismatch pour collection '{collection_name}': existant dim={existing_dim}, requis dim={vector_dim}."
        if not force_recreate:
            raise RuntimeError(msg + " Supprimez la collection ou utilisez --force-recreate.")
        logging.info("force_recreate activé: suppression/recréation.")
        client.delete_collection(collection_name=collection_name)
        client.create_collection(collection_name=collection_name,
                                 vectors_config={vector_name: VectorParams(size=vector_dim, distance=distance)})
        logging.info("Collection recréée '%s' (dim=%d).", collection_name, vector_dim)
    else:
        logging.info("Collection '%s' existe et dimension OK (dim=%d).", collection_name, existing_dim)

# ---------- point id helpers ----------
def point_id_for(pdf_path: Path, page_num: int) -> str:
    key = f"{pdf_path.resolve()}#p{page_num}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

def delete_points_for_pdf(client: QdrantClient, collection_name: str, pdf_path: Path, pages_count: int):
    """
    Supprime les points correspondant aux pages d'un PDF.
    Essaie d'utiliser PointIdsList (forme attendue par les versions récentes du client).
    Fallbacks : appel avec parametre `points=` si présent dans l'API locale.
    Retourne le nombre d'ids supprimés (estimation / best-effort).
    """
    ids = [point_id_for(pdf_path, i+1) for i in range(pages_count)]
    if not ids:
        return 0

    # Assure le bon type : Qdrant attend les mêmes types d'IDs que ceux upsertés (str/int)
    try:
        # Preferred modern API: pass a PointIdsList model
        selector = PointIdsList(points=ids)
        client.delete(collection_name=collection_name, points_selector=selector)
        logging.info("Deleted %d points for %s (PointIdsList).", len(ids), pdf_path)
        return len(ids)
    except Exception as e:
        logging.debug("delete with PointIdsList failed: %s", e)

    # Fallback 1: older client signatures might accept `points=` positional or kwarg
    try:
        # Some qdrant-client versions accept client.delete(collection_name=..., points=[...])
        client.delete(collection_name=collection_name, points=ids)
        logging.info("Deleted %d points for %s (fallback points=).", len(ids), pdf_path)
        return len(ids)
    except Exception as e:
        logging.debug("fallback delete(points=..) failed: %s", e)

    # Fallback 2: positional call (very old signatures)
    try:
        client.delete(collection_name, {"points": ids})
        logging.info("Deleted %d points for %s (fallback positional).", len(ids), pdf_path)
        return len(ids)
    except Exception as e:
        logging.exception("All delete attempts failed for %s : %s", pdf_path, e)
        return 0



# ---------- GPU diagnostics ----------
def gpu_summary():
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        try:
            name = torch.cuda.get_device_name(dev)
        except Exception:
            name = f"cuda:{dev}"
        try:
            allocated = torch.cuda.memory_allocated(dev)
            reserved = torch.cuda.memory_reserved(dev)
        except Exception:
            allocated = reserved = 0
        return {"available": True, "device": "cuda", "device_name": name, "cuda_mem": {"allocated": allocated, "reserved": reserved}}
    else:
        return {"available": False, "device": "cpu"}

# ---------- main per-file processing ----------
def process_pdf_file(pdf_path: Path, client: QdrantClient, embedder: ColPaliEmbedder, collection_name: str,
                     out_image_dir: Optional[Path], batch_size: int, manifest: Dict):
    logging.info("Processing PDF %s", pdf_path)
    t_start_pdf = time.perf_counter()
    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count
    saved_image_dir = None
    if out_image_dir:
        saved_image_dir = out_image_dir
        saved_image_dir.mkdir(parents=True, exist_ok=True)

    points_batch = []
    times = {"render": 0.0, "embed": 0.0, "save": 0.0, "upsert": 0.0}
    pages_processed = 0
    detected_dim = None

    for i in tqdm(range(total_pages), desc="Pages", leave=False):
        page_num_1based = i + 1
        page = doc.load_page(i)

        t0 = time.perf_counter()
        pil_img = render_page_to_pil(page)
        times["render"] += time.perf_counter() - t0

        saved_image_path = None
        if saved_image_dir:
            t0 = time.perf_counter()
            fname = f"{pdf_path.stem}_page_{page_num_1based}.png"
            saved_image_path = str(saved_image_dir / fname)
            try:
                pil_img.save(saved_image_path)
            except Exception:
                logging.exception("Impossible de sauvegarder image %s", saved_image_path)
                saved_image_path = None
            times["save"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            vec = embedder.embed_image(pil_img)
        except Exception:
            logging.exception("Erreur embedding page %d", page_num_1based)
            raise
        times["embed"] += time.perf_counter() - t0

        if detected_dim is None:
            detected_dim = len(vec)
            logging.info("Dimension d'embedding détectée : %d", detected_dim)

        payload = {
            "pdf_filename": pdf_path.name,
            "page_number": page_num_1based,
            "pdf_local_path": str(pdf_path.resolve()),
        }
        if saved_image_path:
            payload["image_path"] = saved_image_path
        try:
            payload["width"], payload["height"] = pil_img.size
        except Exception:
            pass

        pid = point_id_for(pdf_path, page_num_1based)
        vec_list = [float(x) for x in vec]
        point = PointStruct(id=pid, vector={VECTOR_NAME: vec_list}, payload=payload)
        points_batch.append(point)
        pages_processed += 1

        if len(points_batch) >= batch_size:
            t0 = time.perf_counter()
            try:
                client.upsert(collection_name=collection_name, points=points_batch)
            except Exception:
                logging.exception("Upsert failed for batch")
                raise
            times["upsert"] += time.perf_counter() - t0
            points_batch = []

    if points_batch:
        t0 = time.perf_counter()
        client.upsert(collection_name=collection_name, points=points_batch)
        times["upsert"] += time.perf_counter() - t0

    elapsed = time.perf_counter() - t_start_pdf
    logging.info("Finished %s : pages %d processed in %.2fs (render %.2fs embed %.2fs save %.2fs upsert %.2fs).",
                 pdf_path, pages_processed, elapsed, times["render"], times["embed"], times["save"], times["upsert"])

    manifest_entry = {"mtime": pdf_path.lstat().st_mtime, "size": pdf_path.lstat().st_size, "pages": total_pages}
    return manifest_entry, detected_dim

# ---------- top-level: process corpus incrementally ----------
def process_corpus(pdf_dir: Path, qdrant_url: str, api_key: Optional[str],
                   collection_name: str, model_name: str, batch_size: int,
                   out_image_dir: Optional[Path], manifest_path: Path, force_recreate: bool):
    pdf_dir = pdf_dir.resolve()
    manifest = load_manifest(manifest_path)
    pdf_paths = sorted([p for p in pdf_dir.glob("**/*.pdf") if p.is_file()])

    logging.info("PDFs on disk: %d, known in manifest: %d", len(pdf_paths), len(manifest))
    manifest_keys = set(manifest.keys())
    disk_keys = set(str(p.resolve()) for p in pdf_paths)
    deleted = manifest_keys - disk_keys
    added = []
    modified = []
    unchanged = []

    for p in pdf_paths:
        key = str(p.resolve())
        sig = file_signature(p)
        if key not in manifest:
            added.append(p)
        else:
            prev = manifest[key]
            if prev.get("mtime") != sig["mtime"] or prev.get("size") != sig["size"]:
                modified.append(p)
            else:
                unchanged.append(key)

    logging.info("Added: %d, Modified: %d, Deleted: %d, Unchanged: %d", len(added), len(modified), len(deleted), len(unchanged))

    client = QdrantClient(url=qdrant_url, api_key=api_key)
    logging.info("Qdrant ping / summary OK.")
    logging.info("GPU summary: %s", gpu_summary())

    t0 = time.perf_counter()
    embedder = ColPaliEmbedder(model_name=model_name)
    t_embed_init = time.perf_counter() - t0
    logging.info("Embedder initialized (%.3fs).", t_embed_init)

    for key in deleted:
        try:
            prev = manifest[key]
            pages = int(prev.get("pages", 0))
            pdf_path = Path(key)
            delete_points_for_pdf(client, collection_name, pdf_path, pages)
            manifest.pop(key, None)
        except Exception:
            logging.exception("Erreur durant suppression de %s", key)

    collection_ready = False
    detected_dim_global = None

    def ensure_collection_if_needed(detected_dim):
        nonlocal collection_ready, detected_dim_global
        if collection_ready:
            return
        if detected_dim is None:
            return
        ensure_collection_with_dim(client, collection_name, VECTOR_NAME, vector_dim=detected_dim, force_recreate=force_recreate)
        collection_ready = True
        detected_dim_global = detected_dim
        logging.info("Collection ready.")

    for p in added + modified:
        try:
            key = str(p.resolve())
            if key in manifest and key not in [str(x.resolve()) for x in added]:
                prev_pages = int(manifest[key].get("pages", 0))
                if prev_pages > 0:
                    logging.info("File modified: deleting previous %d points for %s", prev_pages, p)
                    delete_points_for_pdf(client, collection_name, p, prev_pages)

            t_file_start = time.perf_counter()
            manifest_entry, detected_dim = process_pdf_file(p, client, embedder, collection_name, out_image_dir, batch_size, manifest)
            t_file_elapsed = time.perf_counter() - t_file_start

            ensure_collection_if_needed(detected_dim)

            manifest[str(p.resolve())] = manifest_entry
            save_manifest(manifest_path, manifest)
            logging.info("Finished %s in %.2fs, manifest updated.", p, t_file_elapsed)
        except Exception:
            logging.exception("Erreur processing %s", p)

    for key in unchanged:
        logging.info("Pas de changement détecté pour %s (skip).", key)

    save_manifest(manifest_path, manifest)
    logging.info("Traitement corpus terminé.")

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Indexer PDF corpus -> ColPali -> Qdrant (incrémental)")
    parser.add_argument("--pdf-dir", required=True, help="Dossier contenant les PDFs")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333", help="URL Qdrant")
    parser.add_argument("--api-key", default=None, help="API key Qdrant")
    parser.add_argument("--collection", default="colpali_pdf_pages_images_incr", help="Nom de la collection Qdrant")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Nom du modèle ColPali")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille de batch upsert")
    parser.add_argument("--out-images", default="./pdf_pages_images", help="Dossier de sauvegarde images (optionnel)")
    parser.add_argument("--manifest", default="./colpali_manifest.json", help="Fichier manifest JSON")
    parser.add_argument("--force-recreate", action="store_true", help="Recréé la collection si dimension mismatch")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        raise SystemExit(f"Répertoire introuvable: {pdf_dir}")

    process_corpus(pdf_dir=pdf_dir,
                   qdrant_url=args.qdrant,
                   api_key=args.api_key,
                   collection_name=args.collection,
                   model_name=args.model,
                   batch_size=args.batch_size,
                   out_image_dir=Path(args.out_images),
                   manifest_path=Path(args.manifest),
                   force_recreate=args.force_recreate)

if __name__ == "__main__":
    main()
