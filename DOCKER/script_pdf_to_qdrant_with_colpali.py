#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script_pdf_to_qdrant_with_colpali.gpucheck.fixed.py

Version corrigée : comparaison des devices robuste (ex: 'cuda' vs 'cuda:0'),
logs de timing, vérifications GPU et upsert dans Qdrant.
"""

import sys
import argparse
import uuid
import traceback
from pathlib import Path
from tqdm import tqdm
import logging
import time
import requests
import json

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import torch

# NOTE: imports ColPali classes selon votre installation transformers
from transformers import ColPaliProcessor, ColPaliForRetrieval

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# ---- logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pdf2qdrant")

# ---------- CONFIG ----------
BATCH_SIZE = 64
RENDER_DPI = 150
VECTOR_NAME = "page_image"
DISTANCE = Distance.COSINE
DEFAULT_COLPALI = "vidore/colpali-v1.3-hf"
# ---------------------------

# ----- GPU helpers -----
def cuda_available():
    return torch.cuda.is_available()

def cuda_device_name():
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        pass
    return None

def cuda_mem_info():
    """Retourne dict avec allocated/reserved (en bytes) si CUDA dispo, sinon None."""
    if not torch.cuda.is_available():
        return None
    try:
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        }
    except Exception:
        return None

def _normalize_device(dev):
    """
    Retourne torch.device à partir d'un input (string, torch.device).
    Si dev est 'cuda' (sans index) on retourne torch.device('cuda') (index=None).
    """
    if isinstance(dev, str):
        try:
            return torch.device(dev)
        except Exception:
            # fallback: if 'cuda' in string but with no index
            if dev.startswith("cuda"):
                return torch.device("cuda")
            return torch.device("cpu")
    if isinstance(dev, torch.device):
        return dev
    # fallback
    return torch.device("cpu")

def device_matches(expected, actual):
    """
    Compare deux torch.device de manière tolérante :
      - types must match (e.g. 'cuda'/'cpu')
      - if expected has no index (e.g. 'cuda') it matches any actual cuda:*
      - if expected has an index, it must equal actual.index
    Retourne True si on considère qu'ils "matchent".
    """
    exp = _normalize_device(expected)
    act = _normalize_device(actual)
    if exp.type != act.type:
        return False
    if exp.type == "cuda":
        # exp.index is None when created from "cuda" (no index) => accept any GPU
        if exp.index is None:
            return True
        return exp.index == act.index
    # for cpu and others, type equality is sufficient
    return True

def find_tensors(obj):
    """Retourne la liste des torch.Tensor trouvés récursivement dans obj."""
    tensors = []
    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            tensors.extend(find_tensors(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            tensors.extend(find_tensors(v))
    return tensors

def assert_tensors_on_device(obj, expected_device, context=""):
    """Vérifie et log si tous les tenseurs dans obj sont sur 'expected_device' (tolérant)."""
    tensors = find_tensors(obj)
    if not tensors:
        logger.debug("GPU CHECK [%s] : pas de tenseurs trouvés à vérifier.", context)
        return True
    ok = True
    for t in tensors:
        if not device_matches(expected_device, t.device):
            logger.warning(
                "GPU CHECK [%s] : tensor device mismatch: expected %s but got %s (shape=%s).",
                context, expected_device, t.device, tuple(t.shape) if hasattr(t, "shape") else None
            )
            ok = False
    if ok:
        logger.debug("GPU CHECK [%s] : tous les tenseurs sont OK sur %s (n=%d).", context, expected_device, len(tensors))
    return ok

# ---------- helpers ----------
def render_page_to_pil(page, dpi=RENDER_DPI):
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def extract_text_from_page(page):
    try:
        return page.get_text("text").strip()
    except Exception:
        return ""

def detect_language_safe(text: str):
    if not text or len(text) < 10:
        return None
    try:
        return detect(text)
    except Exception:
        return None

# ---------- ColPali embedder with GPU checks ----------
class ColPaliEmbedder:
    """
    Embeddeur ColPali -> renvoie une liste 1-D de floats (normalisée).
    Fournit embed_image(img_pil) et vector_dim().
    Ajoute vérifications que les tenseurs sont bien sur le device (CPU/CUDA) attendu.
    """
    def __init__(self, model_name=DEFAULT_COLPALI, device=None):
        # Accept strings like "cuda" or "cuda:0" or torch.device
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = _normalize_device(requested)
        logger.info("Initialisation ColPali '%s' sur device %s.", model_name, self.device)
        if cuda_available():
            logger.info("CUDA disponible : %s", cuda_device_name() or "(nom device inconnu)")

        t0 = time.perf_counter()
        # processor + model
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        # load model and move to device
        self.model = ColPaliForRetrieval.from_pretrained(model_name).to(self.device)
        self.model.eval()
        t1 = time.perf_counter()
        logger.info("Modèle ColPali initialisé en %.3fs.", t1 - t0)

        # quick check model parameters device (tolerant comparison)
        try:
            p = next(self.model.parameters())
            logger.info("Model parameters device: %s", p.device)
            if not device_matches(self.device, p.device):
                logger.warning("Model parameters are not on expected device: expected %s but found %s", self.device, p.device)
        except StopIteration:
            logger.warning("Model has no parameters (?)")

        self._vector_dim = None

    def _extract_embedding_from_output(self, out):
        """
        Tenter d'extraire le tenseur d'embedding depuis la sortie du modèle.
        """
        if out is None:
            raise RuntimeError("Sortie modèle vide.")
        for attr in ("embeddings", "image_embeds", "image_embeds_0", "image_embedding", "pooler_output"):
            if hasattr(out, attr):
                val = getattr(out, attr)
                if val is not None:
                    return val
        if isinstance(out, (tuple, list)) and len(out) > 0:
            return out[0]
        return out

    def embed_image(self, pil_image: Image.Image):
        """Embed une image PIL -> retourne list[float]."""
        # synchronize before measurement if using CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # prepare inputs via processor
        inputs = self.processor(images=pil_image, return_tensors="pt")

        # move tensors to device
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        # check inputs on device (tolerant)
        assert_tensors_on_device(inputs, self.device, context="inputs_before_forward")

        # forward
        with torch.no_grad():
            out = self.model(**inputs)

        # sync
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1_forward = time.perf_counter()

        # check outputs
        ok_out = assert_tensors_on_device(out, self.device, context="model_output")
        if not ok_out:
            logger.debug("Sortie modèle: types/locations des tenseurs potentiellement inattendus (voir warning précédent).")

        emb_candidate = self._extract_embedding_from_output(out)

        # Convert to numpy
        if isinstance(emb_candidate, torch.Tensor):
            emb = emb_candidate.cpu().numpy()
        else:
            emb = np.array(emb_candidate)

        emb = np.asarray(emb)

        # pooling rules -> want 1-D vector
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(axis=0)        # (seq_len, dim)
        if emb.ndim == 2:
            emb = emb.mean(axis=0)
        elif emb.ndim > 2:
            emb = emb.reshape(-1)

        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        # record vector dim
        if self._vector_dim is None:
            self._vector_dim = int(emb.shape[0])

        if self._vector_dim < 1:
            raise RuntimeError(f"Embedding dimension anormale détectée: {self._vector_dim}")

        t1 = time.perf_counter()
        logger.debug("embed_image durations: forward=%.3fs total=%.3fs (dim=%d).",
                     (t1_forward - t0), (t1 - t0), self._vector_dim or -1)
        return emb.tolist()

    def vector_dim(self):
        return self._vector_dim

# ---------- robust ensure_collection_with_dim (copié/ajusté) ----------
def _extract_vectors_from_info_candidate(candidate):
    if not candidate:
        return None
    if isinstance(candidate, dict):
        if "vectors_config" in candidate and isinstance(candidate["vectors_config"], dict):
            return candidate["vectors_config"]
        if "vectors" in candidate and isinstance(candidate["vectors"], dict):
            return candidate["vectors"]
        res = candidate.get("result") or candidate.get("data") or candidate
        if isinstance(res, dict):
            cfg = res.get("config") or res.get("result") or res.get("data") or res
            if isinstance(cfg, dict):
                params = cfg.get("params") or cfg.get("config") or cfg
                if isinstance(params, dict):
                    vecs = params.get("vectors") or params.get("vectors_config")
                    if isinstance(vecs, dict):
                        return vecs
        if "config" in candidate and isinstance(candidate["config"], dict):
            params = candidate["config"].get("params")
            if params and isinstance(params, dict):
                vecs = params.get("vectors")
                if isinstance(vecs, dict):
                    return vecs
    return None

def ensure_collection_with_dim(client: QdrantClient, collection_name: str,
                               vector_name: str, vector_dim: int, force_recreate: bool=False,
                               qdrant_url: str = None, api_key: str = None):
    try:
        exists = client.collection_exists(collection_name=collection_name)
    except Exception as e:
        raise RuntimeError(f"Erreur connexion Qdrant: {e}")

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={vector_name: VectorParams(size=vector_dim, distance=DISTANCE)}
        )
        logger.info("Collection '%s' créée (vector %s, dim=%d).", collection_name, vector_name, vector_dim)
        return

    existing_vectors = None

    # 1) get_collection
    try:
        info = client.get_collection(collection_name=collection_name)
        if not isinstance(info, dict):
            try:
                info = info.__dict__
            except Exception:
                pass
        existing_vectors = _extract_vectors_from_info_candidate(info)
    except Exception:
        logger.debug("client.get_collection() a levé, on continue les fallbacks.")

    # 2) get_collections
    if not existing_vectors:
        try:
            all_info = client.get_collections()
            cols = getattr(all_info, "collections", None) or (all_info.get("result", {}).get("collections") if isinstance(all_info, dict) else None) or all_info
            if isinstance(cols, (list, tuple)):
                for col in cols:
                    name = getattr(col, "name", None) or (col.get("name") if isinstance(col, dict) else None)
                    if name == collection_name:
                        cand = col
                        if not isinstance(cand, dict):
                            try:
                                cand = cand.__dict__
                            except Exception:
                                pass
                        existing_vectors = _extract_vectors_from_info_candidate(cand)
                        if existing_vectors:
                            break
        except Exception:
            logger.debug("client.get_collections() a levé, on continue les fallbacks.")

    # 3) HTTP /collections/{name}
    if not existing_vectors and qdrant_url:
        try:
            url = qdrant_url.rstrip("/") + f"/collections/{collection_name}"
            headers = {}
            if api_key:
                headers["api-key"] = api_key
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.ok:
                j = resp.json()
                existing_vectors = _extract_vectors_from_info_candidate(j)
            else:
                logger.debug("GET %s -> status %d", url, resp.status_code)
        except Exception as e:
            logger.debug("Requête HTTP vers /collections/{name} échouée: %s", e)

    # 4) HTTP /collections
    if not existing_vectors and qdrant_url:
        try:
            url_all = qdrant_url.rstrip("/") + "/collections"
            headers = {}
            if api_key:
                headers["api-key"] = api_key
            r = requests.get(url_all, headers=headers, timeout=10)
            if r.ok:
                j = r.json()
                cand_cols = j.get("result", {}).get("collections") or j.get("collections") or j.get("result")
                if isinstance(cand_cols, list):
                    for col in cand_cols:
                        name = col.get("name") if isinstance(col, dict) else getattr(col, "name", None)
                        if name == collection_name:
                            existing_vectors = _extract_vectors_from_info_candidate(col)
                            if existing_vectors:
                                break
                else:
                    existing_vectors = _extract_vectors_from_info_candidate(j)
            else:
                logger.debug("GET %s -> status %d", url_all, r.status_code)
        except Exception as e:
            logger.debug("GET /collections échouée: %s", e)

    if not existing_vectors:
        logger.error("Impossible d'extraire vectors_config; dump minimal pour debug (si disponible).")
        if qdrant_url:
            try:
                headers = {}
                if api_key:
                    headers["api-key"] = api_key
                r = requests.get(qdrant_url.rstrip("/") + "/collections", headers=headers, timeout=10)
                if r.ok:
                    logger.error(json.dumps(r.json(), indent=2)[:2000])
            except Exception:
                pass
        raise RuntimeError(
            "Impossible de lire vectors_config pour la collection existante. "
            "Vérifiez la version du client qdrant et l'API."
        )

    # existing_vectors is dict
    if vector_name in existing_vectors:
        existing_dim = int(existing_vectors[vector_name].get("size") or existing_vectors[vector_name].get("dimension") or existing_vectors[vector_name].get("dim"))
    else:
        first_key = next(iter(existing_vectors.keys()))
        existing_dim = int(existing_vectors[first_key].get("size") or existing_vectors[first_key].get("dimension") or existing_vectors[first_key].get("dim"))
        logger.warning("Le vector_name '%s' n'a pas été trouvé ; utilisation de '%s' (dim=%d).", vector_name, first_key, existing_dim)

    if existing_dim != vector_dim:
        msg = (f"Dimension mismatch pour collection '{collection_name}': existant dim={existing_dim}, requis dim={vector_dim}.")
        if not force_recreate:
            msg += " Supprimez la collection, utilisez un nom différent ou relancez avec --force-recreate."
            raise RuntimeError(msg)
        else:
            logger.info("force_recreate activé : suppression/recréation (destructif).")
            client.delete_collection(collection_name=collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config={vector_name: VectorParams(size=vector_dim, distance=DISTANCE)}
            )
            logger.info("Collection '%s' recréée (dim=%d).", collection_name, vector_dim)
    else:
        logger.info("Collection '%s' existe et dimension OK (dim=%d).", collection_name, existing_dim)

# ---------- main processing ----------
def process_pdf_to_qdrant(pdf_path: Path, qdrant_url: str, api_key: str, collection_name: str,
                          out_image_dir: Path = None, max_pages: int = None,
                          model_name: str = DEFAULT_COLPALI, force_recreate: bool=False):
    total_start = time.perf_counter()

    logger.info("GPU summary: available=%s, cuda_device_name=%s, cuda_mem=%s",
                cuda_available(), cuda_device_name(), cuda_mem_info())

    client = QdrantClient(url=qdrant_url, api_key=api_key)

    # init embedder (time logged inside)
    t0 = time.perf_counter()
    embedder = ColPaliEmbedder(model_name=model_name)
    t1 = time.perf_counter()
    logger.info("Temps initialisation embedder: %.3fs", t1 - t0)

    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
    if total_pages == 0:
        logger.warning("Le PDF contient 0 page.")
        return

    if out_image_dir:
        out_image_dir.mkdir(parents=True, exist_ok=True)

    points_batch = []
    collection_ready = False
    upsert_total_time = 0.0
    embedding_total_time = 0.0
    render_total_time = 0.0
    save_total_time = 0.0

    for page_number in tqdm(range(total_pages), desc="Pages"):
        page_proc_start = time.perf_counter()

        page = doc.load_page(page_number)
        page_num_1based = page_number + 1

        # render
        if embedder.device.type == "cuda":
            torch.cuda.synchronize()
        t_render0 = time.perf_counter()
        pil_img = render_page_to_pil(page)
        if embedder.device.type == "cuda":
            torch.cuda.synchronize()
        t_render1 = time.perf_counter()
        render_dur = t_render1 - t_render0
        render_total_time += render_dur
        logger.debug("Page %d: render time %.3fs", page_num_1based, render_dur)

        # extract native text (quick)
        text = extract_text_from_page(page)
        lang = detect_language_safe(text) if text else None

        # optional save image
        saved_image_path = None
        if out_image_dir:
            if embedder.device.type == "cuda":
                torch.cuda.synchronize()
            t_save0 = time.perf_counter()
            img_name = f"{pdf_path.stem}_page_{page_num_1based}.png"
            saved_image_path = str(out_image_dir / img_name)
            try:
                pil_img.save(saved_image_path)
            except Exception as e:
                logger.warning("Impossible de sauvegarder l'image %s : %s", saved_image_path, e)
                saved_image_path = None
            if embedder.device.type == "cuda":
                torch.cuda.synchronize()
            t_save1 = time.perf_counter()
            save_dur = t_save1 - t_save0
            save_total_time += save_dur
            logger.debug("Page %d: save image time %.3fs", page_num_1based, save_dur)

        # embedding
        if embedder.device.type == "cuda":
            torch.cuda.synchronize()
        t_emb0 = time.perf_counter()
        try:
            vec = embedder.embed_image(pil_img)
        except Exception as e:
            logger.error("Erreur embedding page %d : %s", page_num_1based, e)
            raise
        if embedder.device.type == "cuda":
            torch.cuda.synchronize()
        t_emb1 = time.perf_counter()
        emb_dur = t_emb1 - t_emb0
        embedding_total_time += emb_dur
        logger.info("Page %d: embedding time %.3fs (dim=%d)", page_num_1based, emb_dur, len(vec))

        # ensure collection after having a real dim
        if not collection_ready:
            detected_dim = len(vec)
            logger.info("Dimension d'embedding détectée : %d", detected_dim)
            t_col0 = time.perf_counter()
            ensure_collection_with_dim(client, collection_name, VECTOR_NAME, vector_dim=detected_dim, force_recreate=force_recreate, qdrant_url=qdrant_url, api_key=api_key)
            t_col1 = time.perf_counter()
            logger.info("Collection ready (temps création/check %.3fs).", t_col1 - t_col0)
            collection_ready = True

        payload = {
            "pdf_filename": pdf_path.name,
            "page_number": page_num_1based,
            "text": text,
            "lang": lang,
        }
        if saved_image_path:
            payload["image_path"] = saved_image_path
        try:
            payload["width"], payload["height"] = pil_img.size
        except Exception:
            pass

        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{pdf_path.resolve()}#p{page_num_1based}"))
        point = PointStruct(id=point_id, vector={VECTOR_NAME: vec}, payload=payload)
        points_batch.append(point)

        # upsert batch if full
        if len(points_batch) >= BATCH_SIZE:
            t_up0 = time.perf_counter()
            logger.info("Upserting batch de %d points...", len(points_batch))
            try:
                client.upsert(collection_name=collection_name, points=points_batch)
            except Exception:
                traceback.print_exc()
                logger.error("Échec upsert pour batch. IDs: %s", [p.id for p in points_batch])
                raise
            t_up1 = time.perf_counter()
            upsert_dur = t_up1 - t_up0
            upsert_total_time += upsert_dur
            logger.info("Batch upserted en %.3fs", upsert_dur)
            points_batch = []

        page_proc_end = time.perf_counter()
        logger.debug("Page %d total page processing time: %.3fs", page_num_1based, page_proc_end - page_proc_start)
        # log CUDA mem snapshot per page (debug)
        mem = cuda_mem_info()
        if mem:
            logger.debug("CUDA mem after page %d: %s", page_num_1based, mem)

    # final batch
    if points_batch:
        t_up0 = time.perf_counter()
        logger.info("Upserting batch final de %d points...", len(points_batch))
        try:
            client.upsert(collection_name=collection_name, points=points_batch)
        except Exception:
            traceback.print_exc()
            logger.error("Échec upsert final. IDs: %s", [p.id for p in points_batch])
            raise
        t_up1 = time.perf_counter()
        upsert_dur = t_up1 - t_up0
        upsert_total_time += upsert_dur
        logger.info("Batch final upserted en %.3fs", upsert_dur)

    total_end = time.perf_counter()
    total_elapsed = total_end - total_start

    logger.info("=== Résumé des durées ===")
    logger.info("Pages traitées : %d", total_pages)
    logger.info("Temps total exécution : %.3fs", total_elapsed)
    logger.info(" - Temps initialisation embedder : %.3fs", t1 - t0)
    logger.info(" - Temps rendu total (pages) : %.3fs", render_total_time)
    logger.info(" - Temps sauvegarde images total : %.3fs", save_total_time)
    logger.info(" - Temps embedding total : %.3fs", embedding_total_time)
    logger.info(" - Temps upsert total : %.3fs", upsert_total_time)
    logger.info("=== Fin ===")

def main():
    parser = argparse.ArgumentParser(description="Indexer pages PDF -> ColPali -> Qdrant (avec GPU checks & timings)")
    parser.add_argument("--pdf", required=True, help="Chemin vers le fichier PDF")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333", help="URL Qdrant")
    parser.add_argument("--api-key", default=None, help="API key Qdrant")
    parser.add_argument("--collection", default="colpali_pdf_pages_images", help="Nom collection Qdrant")
    parser.add_argument("--out-images", default="./pdf_pages_images", help="Dossier pour sauvegarder images")
    parser.add_argument("--max-pages", type=int, default=None, help="Limiter nombre de pages")
    parser.add_argument("--model", default=DEFAULT_COLPALI, help="Nom du modèle ColPali")
    parser.add_argument("--force-recreate", action="store_true", help="Supprimer/recréer collection en cas de mismatch (destructif)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"Fichier PDF introuvable : {pdf_path}")

    process_pdf_to_qdrant(pdf_path, args.qdrant, args.api_key, args.collection,
                          out_image_dir=Path(args.out_images), max_pages=args.max_pages,
                          model_name=args.model, force_recreate=args.force_recreate)

if __name__ == "__main__":
    main()
