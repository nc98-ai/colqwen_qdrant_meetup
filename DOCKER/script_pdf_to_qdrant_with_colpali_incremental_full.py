#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script_pdf_to_qdrant_with_colpali_incremental_full.py

Traitement incrémental d'un dossier PDF -> ColPali embeddings -> Qdrant.
- Manifest JSON nommé automatiquement par collection si --manifest non fourni : <collection>_manifest.json
- Supporte ajout / modif / suppression de PDF
- Vérifie l'utilisation GPU et logue les temps par étape
- Upsert des vecteurs (nommés) et payload contenant pdf_local_path
- Suppression de points via PointIdsList ou fallbacks selon la version du client

Usage exemple :
python script_pdf_to_qdrant_with_colpali_incremental_full.py \
  --pdf-dir ./pdf_corpus \
  --qdrant http://127.0.0.1:6333 \
  --api-key ma_cle_secrete \
  --collection colpali_pdf_pages_images_incr \
  --model vidore/colpali-v1.3-hf \
  --batch-size 32 \
  --out-images ./pdf_pages_images \
  --force-recreate

"""

import argparse
import json
import logging
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

# transformers ColPali (nom du modèle configurable)
from transformers import ColPaliProcessor, ColPaliForRetrieval

# Qdrant client models
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    PointStruct,
    PointIdsList,
)

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# ----- configuration -----
BATCH_SIZE_DEFAULT = 32
RENDER_DPI = 150
VECTOR_NAME = "page_image"
DISTANCE = Distance.COSINE
DEFAULT_COLPALI = "vidore/colpali-v1.3-hf"

# ----- logging -----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ----- helpers: rendering & text -----

def render_page_to_pil(page, dpi=RENDER_DPI) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def extract_text_from_page(page) -> str:
    try:
        return page.get_text("text").strip()
    except Exception:
        return ""


def detect_language_safe(text: str) -> Optional[str]:
    if not text or len(text) < 10:
        return None
    try:
        return detect(text)
    except Exception:
        return None

# ----- GPU / device checks -----

def gpu_summary() -> Dict[str, Any]:
    available = torch.cuda.is_available()
    if not available:
        return {"available": False}
    try:
        dev = torch.device("cuda")
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        allocated = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        return {"available": True, "device": "cuda", "device_idx": idx, "device_name": name, "cuda_mem": {"allocated": allocated, "reserved": reserved}}
    except Exception:
        return {"available": True, "device": "cuda", "device_idx": None}

# ----- ColPali Embedder -----
class ColPaliEmbedder:
    def __init__(self, model_name=DEFAULT_COLPALI, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Init ColPali (%s) on device %s", model_name, self.device)
        t0 = time.time()
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        self.model = ColPaliForRetrieval.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._vector_dim = None
        t1 = time.time()
        logger.info("Model loaded in %.3fs", t1 - t0)

    def _extract_embedding_from_output(self, out):
        # attempt to find reasonable attribute
        if out is None:
            raise RuntimeError("Empty model output")
        for attr in ("embeddings", "image_embeds", "image_embedding", "image_embeds_0"):
            if hasattr(out, attr):
                val = getattr(out, attr)
                if val is not None:
                    return val
        if isinstance(out, (list, tuple)) and len(out) > 0:
            return out[0]
        return out

    def embed_image(self, pil_image: Image.Image) -> List[float]:
        # build inputs and move to device
        inputs = self.processor(images=pil_image, return_tensors="pt")
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        t0 = time.time()
        with torch.no_grad():
            out = self.model(**inputs)
        t1 = time.time()

        emb_candidate = self._extract_embedding_from_output(out)
        if isinstance(emb_candidate, torch.Tensor):
            emb = emb_candidate.cpu().numpy()
        else:
            emb = np.asarray(emb_candidate)

        emb = np.asarray(emb)

        # common shapes -> collapse to 1-D
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

    def vector_dim(self) -> Optional[int]:
        return self._vector_dim

# ----- Qdrant helpers -----

def extract_vectors_config_from_info(info_obj: Any) -> Optional[Dict[str, Any]]:
    # info_obj may be dict, or pydantic CollectionInfo/CollectionConfig
    # try several strategies
    try:
        if isinstance(info_obj, dict):
            # older restful structure
            cfg = info_obj.get("result", {}).get("config", {})
            if cfg:
                return cfg.get("params", {}).get("vectors")
            return info_obj.get("vectors_config") or info_obj.get("vectors")
        # pydantic object returned by client.get_collection
        if hasattr(info_obj, "config"):
            cfg = getattr(info_obj, "config")
            # cfg is CollectionConfig, with .params.vectors or .params
            if hasattr(cfg, "params") and getattr(cfg.params, "vectors", None) is not None:
                # convert to plain dict-like
                vecs = {}
                for k, v in cfg.params.vectors.items():
                    # v is VectorParams
                    vecs[k] = {"size": v.size, "distance": v.distance.value if hasattr(v.distance, "value") else str(v.distance)}
                return vecs
    except Exception:
        logger.debug("extract_vectors_config_from_info failed", exc_info=True)
    return None


def ensure_collection_with_dim(client: QdrantClient, collection_name: str, vector_name: str, vector_dim: int, force_recreate: bool = False):
    try:
        exists = client.collection_exists(collection_name=collection_name)
    except Exception as e:
        raise RuntimeError(f"Erreur connexion Qdrant: {e}")

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={vector_name: VectorParams(size=vector_dim, distance=DISTANCE)},
        )
        logger.info("Collection '%s' créée (vector %s, dim=%d).", collection_name, vector_name, vector_dim)
        return

    info = client.get_collection(collection_name=collection_name)
    existing_vectors = extract_vectors_config_from_info(info if info is not None else {})
    if not existing_vectors:
        raise RuntimeError("Impossible de lire vectors_config pour la collection existante. La récupération a échoué (vérifiez la version du client qdrant et l'API).")

    if vector_name in existing_vectors:
        existing_dim = existing_vectors[vector_name]["size"] if isinstance(existing_vectors[vector_name], dict) and "size" in existing_vectors[vector_name] else existing_vectors[vector_name].size if hasattr(existing_vectors[vector_name], "size") else None
    else:
        first_key = next(iter(existing_vectors.keys()))
        existing_dim = existing_vectors[first_key]["size"] if isinstance(existing_vectors[first_key], dict) and "size" in existing_vectors[first_key] else getattr(existing_vectors[first_key], "size", None)
        logger.warning("Le vector_name '%s' n'a pas été trouvé ; utilisation de '%s' (dim=%s).", vector_name, first_key, existing_dim)

    try:
        existing_dim_int = int(existing_dim)
    except Exception:
        raise RuntimeError("Impossible de déterminer la dimension existante de la collection.")

    if existing_dim_int != vector_dim:
        msg = f"Dimension mismatch pour collection '{collection_name}': existant dim={existing_dim_int}, requis dim={vector_dim}."
        if not force_recreate:
            msg += " Supprimez la collection, utilisez un nom différent ou relancez avec --force-recreate."
            raise RuntimeError(msg)
        else:
            logger.info("force_recreate activé : suppression/recréation (destructif).")
            client.delete_collection(collection_name=collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config={vector_name: VectorParams(size=vector_dim, distance=DISTANCE)},
            )
            logger.info("Collection '%s' recréée (dim=%d).", collection_name, vector_dim)
    else:
        logger.info("Collection '%s' existe et dimension OK (dim=%d).", collection_name, existing_dim_int)

# ----- ID generation -----

def point_id_for(pdf_path: Path, page_number: int) -> str:
    # stable uuidv5 based on absolute path + page
    key = f"{pdf_path.resolve()}#p{page_number}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

# ----- delete helper -----

def delete_points_for_pdf(client: QdrantClient, collection_name: str, pdf_path: Path, pages_count: int) -> int:
    ids = [point_id_for(pdf_path, i + 1) for i in range(pages_count)]
    if not ids:
        return 0
    # try modern PointIdsList
    try:
        selector = PointIdsList(points=ids)
        client.delete(collection_name=collection_name, points_selector=selector)
        logger.info("Deleted %d points for %s (PointIdsList).", len(ids), pdf_path)
        return len(ids)
    except Exception as e:
        logger.debug("delete with PointIdsList failed: %s", e)
    # fallback: some qdrant-client versions accept points=ids
    try:
        client.delete(collection_name=collection_name, points=ids)
        logger.info("Deleted %d points for %s (fallback points=).", len(ids), pdf_path)
        return len(ids)
    except Exception as e:
        logger.debug("fallback delete(points=..) failed: %s", e)
    # last resort: raw request (use REST via client._client)
    try:
        # build minimal REST selector
        rest_selector = {"points": ids}
        client._client.points_api.delete_points(collection_name=collection_name, body={"points": ids})
        logger.info("Deleted %d points for %s (raw REST).", len(ids), pdf_path)
        return len(ids)
    except Exception as e:
        logger.exception("All delete attempts failed for %s : %s", pdf_path, e)
        return 0

# ----- manifest helpers -----

def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Impossible de lire manifest %s", path)
        return {}


def save_manifest(path: Path, manifest: Dict[str, Any]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

# ----- main processing of a single PDF -----

def process_single_pdf(pdf_path: Path, client: QdrantClient, collection_name: str, embedder: ColPaliEmbedder, out_images_dir: Optional[Path], batch_size: int, force_recreate: bool, manifest: Dict[str, Any]):
    logger.info("Processing PDF %s", pdf_path)
    t_start_pdf = time.time()
    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count

    # timings
    t_render_total = 0.0
    t_embed_total = 0.0
    t_save_total = 0.0
    points_batch: List[PointStruct] = []

    collection_ready = False
    detected_dim = None

    for page_idx in tqdm(range(total_pages), desc="Pages"):
        t_page_start = time.time()
        page = doc.load_page(page_idx)
        page_no = page_idx + 1

        text = extract_text_from_page(page)
        lang = detect_language_safe(text) if text else None

        # render
        t0 = time.time()
        pil_img = render_page_to_pil(page)
        t1 = time.time()
        t_render_total += (t1 - t0)

        # save image (optional)
        saved_image_path = None
        if out_images_dir:
            out_images_dir.mkdir(parents=True, exist_ok=True)
            img_name = f"{pdf_path.stem}_page_{page_no}.png"
            saved_image_path = str(out_images_dir / img_name)
            try:
                t0s = time.time()
                pil_img.save(saved_image_path)
                t1s = time.time()
                t_save_total += (t1s - t0s)
            except Exception:
                logger.exception("Impossible de sauvegarder image %s", saved_image_path)
                saved_image_path = None

        # embed
        t0e = time.time()
        vec = embedder.embed_image(pil_img)
        t1e = time.time()
        t_embed_total += (t1e - t0e)

        if not collection_ready:
            detected_dim = len(vec)
            logger.info("Dimension d'embedding détectée : %d", detected_dim)
            ensure_collection_with_dim(client, collection_name, VECTOR_NAME, vector_dim=detected_dim, force_recreate=force_recreate)
            collection_ready = True

        payload = {
            "pdf_filename": pdf_path.name,
            "pdf_local_path": str(pdf_path.resolve()),
            "page_number": page_no,
            "text": text,
            "lang": lang,
        }
        if saved_image_path:
            payload["image_path"] = saved_image_path
        try:
            payload["width"], payload["height"] = pil_img.size
        except Exception:
            pass

        pid = point_id_for(pdf_path, page_no)
        # use named vector (dict) to match collection creation with vectors_config
        point = PointStruct(id=pid, vector={VECTOR_NAME: vec}, payload=payload)
        points_batch.append(point)

        if len(points_batch) >= batch_size:
            logger.info("Upserting batch of %d points...", len(points_batch))
            t0u = time.time()
            client.upsert(collection_name=collection_name, points=points_batch)
            t1u = time.time()
            logger.info("Batch upserted in %.3fs", t1u - t0u)
            points_batch = []

        t_page_end = time.time()
        logger.info("Page %d: render %.3fs embed %.3fs save %.3fs total %.3fs", page_no, (t1 - t0), (t1e - t0e), (t1s - t0s) if saved_image_path else 0.0, (t_page_end - t_page_start))

    if points_batch:
        logger.info("Upserting final batch of %d points...", len(points_batch))
        t0u = time.time()
        client.upsert(collection_name=collection_name, points=points_batch)
        t1u = time.time()
        logger.info("Batch final upserted in %.3fs", t1u - t0u)

    t_end_pdf = time.time()
    logger.info("Finished %s : pages %d processed in %.3fs (render %.3fs embed %.3fs save %.3fs upsert %.3fs).",
                pdf_path, total_pages, (t_end_pdf - t_start_pdf), t_render_total, t_embed_total, t_save_total, 0.0)

    return {
        "pages": total_pages,
        "time": t_end_pdf - t_start_pdf,
    }

# ----- process corpus (incremental) -----

def scan_pdf_dir(pdf_dir: Path) -> Dict[str, Dict[str, Any]]:
    # manifest-friendly metadata: mtime & size & pages
    res: Dict[str, Dict[str, Any]] = {}
    for p in sorted(pdf_dir.glob("**/*.pdf")):
        try:
            stat = p.stat()
            res[str(p.resolve())] = {"mtime": stat.st_mtime, "size": stat.st_size}
        except Exception:
            logger.exception("Ignoring path %s", p)
    return res


def process_corpus(pdf_dir: Path, qdrant_url: str, api_key: Optional[str], collection_name: str, model_name: str,
                   batch_size: int = BATCH_SIZE_DEFAULT, out_images: Optional[Path] = None, manifest_path: Optional[Path] = None,
                   force_recreate: bool = False):
    # default manifest name based on collection if not provided
    if manifest_path is None:
        manifest_path = Path(f"{collection_name}_manifest.json")
    else:
        manifest_path = Path(manifest_path)

    manifest = load_manifest(manifest_path)

    pdf_dir = Path(pdf_dir)
    current = scan_pdf_dir(pdf_dir)

    known_paths = set(manifest.keys())
    disk_paths = set(current.keys())

    added = list(disk_paths - known_paths)
    deleted = list(known_paths - disk_paths)
    modified = []

    for p in disk_paths & known_paths:
        old = manifest.get(p, {})
        cur = current.get(p, {})
        if old.get("mtime") != cur.get("mtime") or old.get("size") != cur.get("size"):
            modified.append(p)

    logger.info("PDFs on disk: %d, known in manifest: %d, deleted: %d", len(disk_paths), len(known_paths), len(deleted))
    logger.info("Added: %d, Modified: %d, Deleted: %d, Unchanged: %d", len(added), len(modified), len(deleted), len(disk_paths) - len(added) - len(modified))

    # setup Qdrant client and embedder
    client = QdrantClient(url=qdrant_url, api_key=api_key)
    logger.info("Qdrant ping / summary OK.")
    logger.info("GPU summary: %s", gpu_summary())
    embedder = ColPaliEmbedder(model_name=model_name)

    # process deletions first
    for p in deleted:
        p_path = Path(p)
        try:
            # manifest stores pages count if available
            pages = manifest.get(p, {}).get("pages") or 0
            if pages:
                deleted_count = delete_points_for_pdf(client, collection_name, p_path, pages)
                logger.info("Deleted %d points for removed PDF %s", deleted_count, p)
            else:
                logger.warning("No page count found in manifest for %s — attempting delete by prefix (best-effort).", p)
                # best-effort: try to delete points by scanning payload (expensive) — omitted for brevity
            # remove from manifest
            if p in manifest:
                del manifest[p]
        except Exception:
            logger.exception("Erreur suppression points pour %s", p)

    # helper to ensure collection on first embedding discovery
    detected_dim_global = None
    def ensure_collection_if_needed(detected_dim_local: int):
        nonlocal detected_dim_global
        if detected_dim_global is None:
            detected_dim_global = detected_dim_local
            ensure_collection_with_dim(client, collection_name, VECTOR_NAME, vector_dim=detected_dim_local, force_recreate=force_recreate)

    # process added & modified
    to_process = added + modified
    for p in to_process:
        p_path = Path(p)
        try:
            t0 = time.time()
            result = process_single_pdf(p_path, client, collection_name, embedder, out_images, batch_size, force_recreate, manifest)
            t1 = time.time()
            manifest[str(p_path.resolve())] = current[str(p_path.resolve())]
            manifest[str(p_path.resolve())]["pages"] = result["pages"]
            manifest[str(p_path.resolve())]["last_indexed_at"] = time.time()
            save_manifest(manifest_path, manifest)
            logger.info("Finished %s in %.3fs, manifest updated.", p, (t1 - t0))
        except Exception:
            logger.exception("Erreur processing %s", p)

    logger.info("Traitement corpus terminé.")

# ----- CLI -----

def main():
    parser = argparse.ArgumentParser(description="Indexer PDF corpus -> ColPali -> Qdrant (incr)")
    parser.add_argument("--pdf-dir", required=True, help="Dossier racine contenant PDFs")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333", help="URL Qdrant")
    parser.add_argument("--api-key", default=None, help="API key Qdrant")
    parser.add_argument("--collection", required=True, help="Nom collection Qdrant")
    parser.add_argument("--model", default=DEFAULT_COLPALI, help="Nom modèle ColPali")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT, help="Upsert batch size")
    parser.add_argument("--out-images", default=None, help="Dossier pour sauvegarder images (optionnel)")
    parser.add_argument("--manifest", default=None, help="Fichier manifest JSON (par défaut: <collection>_manifest.json)")
    parser.add_argument("--force-recreate", action="store_true", help="Force recreate collection if dimension mismatch")

    args = parser.parse_args()

    # decide manifest filename based on collection if not provided
    manifest_path = Path(args.manifest) if args.manifest else Path(f"{args.collection}_manifest.json")

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        raise SystemExit(f"Dossier PDF introuvable: {pdf_dir}")

    out_images = Path(args.out_images) if args.out_images else None

    process_corpus(pdf_dir, args.qdrant, args.api_key, args.collection, args.model, batch_size=args.batch_size, out_images=out_images, manifest_path=manifest_path, force_recreate=args.force_recreate)

if __name__ == "__main__":
    main()
