#!/usr/bin/env python3
"""
script_pdf_to_qdrant_with_tesseract.py

Extrait les pages d'un PDF -> image -> embeddings CLIP -> upsert dans Qdrant.
Usage:
python script_pdf_to_qdrant_with_tesseract.py --pdf trunksip.pdf --qdrant http://127.0.0.1:6333 --api-key ma_cle_secrete
"""

import sys
import os
import argparse
import uuid
import traceback
from pathlib import Path
from tqdm import tqdm

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import torch

from transformers import CLIPProcessor, CLIPModel

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

import pytesseract
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# Debug info : quel fichier est exécuté
print("DEBUG: Running script file:", __file__, " (python executable:", sys.executable, ")")

# ---------- CONFIG ----------
BATCH_SIZE = 32        # upsert par lot (reduire si mémoire limitée)
RENDER_DPI = 150       # qualité rendu page -> image (200-300 si OCR fin)
VECTOR_NAME = "page_image"
VECTOR_DIM = 512       # CLIP ViT-B/32 => 512
DISTANCE = Distance.COSINE
DEFAULT_MODEL = "openai/clip-vit-base-patch32"

# ---------- helpers ----------
def render_page_to_pil(page, dpi=RENDER_DPI):
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return img

def extract_text_from_page(page):
    return page.get_text("text").strip()

def ocr_image(image: Image.Image, lang_hint=None):
    langs = []
    if lang_hint and lang_hint.startswith("fr"):
        langs = ["fra", "eng"]
    elif lang_hint and lang_hint.startswith("es"):
        langs = ["spa", "eng"]
    else:
        langs = ["eng", "fra", "spa"]
    lang_param = "+".join(langs)
    try:
        txt = pytesseract.image_to_string(image, lang=lang_param)
    except Exception:
        txt = pytesseract.image_to_string(image)
    return txt.strip()

def detect_language(text: str):
    if not text or len(text) < 10:
        return None
    try:
        return detect(text)
    except Exception:
        return None

# ---------- CLIP embedder ----------
class ClipEmbedder:
    def __init__(self, model_name=DEFAULT_MODEL, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing CLIP model '{model_name}' on device {self.device} (this can download ~600MB).")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_image(self, pil_image: Image.Image):
        # processor will convert pil to tensors
        inputs = self.processor(images=pil_image, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        vec = image_features.cpu().numpy()[0]
        # normaliser (recommandé pour Cosine)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

# ---------- Qdrant helpers ----------
def ensure_collection(client: QdrantClient, collection_name: str):
    try:
        exists = client.collection_exists(collection_name=collection_name)
    except Exception as e:
        raise RuntimeError(f"Erreur connexion Qdrant: {e}")
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={VECTOR_NAME: VectorParams(size=VECTOR_DIM, distance=DISTANCE)}
        )
        print(f"Collection '{collection_name}' créée (vector {VECTOR_NAME}, dim={VECTOR_DIM}).")
    else:
        print(f"Collection '{collection_name}' existe déjà.")

# ---------- main processing ----------
def process_pdf_to_qdrant(pdf_path: Path, qdrant_url: str, api_key: str, collection_name: str,
                          out_image_dir: Path = None, max_pages: int = None, model_name=DEFAULT_MODEL):
    # client Qdrant
    client = QdrantClient(url=qdrant_url, api_key=api_key)
    ensure_collection(client, collection_name)

    # embedder CLIP
    embedder = ClipEmbedder(model_name=model_name)

    # open PDF
    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count if max_pages is None else min(doc.page_count, max_pages)

    if out_image_dir:
        out_image_dir.mkdir(parents=True, exist_ok=True)

    points_batch = []
    for page_number in tqdm(range(total_pages), desc="Pages"):
        page = doc.load_page(page_number)
        page_num_1based = page_number + 1

        # extract text natif
        text = extract_text_from_page(page)
        lang = detect_language(text) if text else None

        # render image
        pil_img = render_page_to_pil(page)

        # OCR si nécessaire
        if not text:
            ocr_text = ocr_image(pil_img, lang_hint=lang)
            text = ocr_text
            lang = detect_language(text) or lang

        # optional save of image
        saved_image_path = None
        if out_image_dir:
            img_name = f"{pdf_path.stem}_page_{page_num_1based}.png"
            saved_image_path = str(out_image_dir / img_name)
            try:
                pil_img.save(saved_image_path)
            except Exception as e:
                print(f"Warning: impossible de sauvegarder l'image {saved_image_path}: {e}")
                saved_image_path = None

        # embedding CLIP
        try:
            vec = embedder.embed_image(pil_img)
        except Exception as e:
            raise RuntimeError(f"Erreur embedding page {page_num_1based}: {e}")

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
        
        import zlib
        def id_crc32_for_page(pdf_path: Path, page_num: int) -> int:
            key = f"{pdf_path.resolve()}#p{page_num}".encode("utf-8")
            return zlib.crc32(key) & 0xFFFFFFFF  # assure unsigned 32-bit        
        
        import hashlib     
        def blake2b64_normalized_pixels(path: Path, size=(512,512)) -> int:
            img = Image.open(path).convert("RGB")
            img = img.resize(size, resample=Image.LANCZOS)   # normaliser taille
            data = img.tobytes()                             # octets pixels RGB continus
            h = hashlib.blake2b(data, digest_size=8)
            return int.from_bytes(h.digest(), "big") 
        
        def blake2b64_normalized_pixels(path: Path, size=(512,512)) -> int:
            img = Image.open(path).convert("RGB")
            img = img.resize(size, resample=Image.LANCZOS)   # normaliser taille
            data = img.tobytes()                             # octets pixels RGB continus
            h = hashlib.blake2b(data, digest_size=8)
            return int.from_bytes(h.digest(), "big")        

        # --- generate stable UUID string for point id (UUIDv5) ---
        point_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"{pdf_path.resolve()}#p{page_num_1based}")
        # point_id = str(point_uuid)   # IMPORTANT: string UUID
        # point_id = id_crc32_for_page(pdf_path, page_num_1based)
        # point_id = id_blake2b64_for_page(pdf_path, page_num_1based)
        point_id = blake2b64_normalized_pixels(saved_image_path, size=(512,512))

        point = PointStruct(id=point_id, vector={VECTOR_NAME: vec}, payload=payload)
        points_batch.append(point)

        # debug: show prepared id
        print(f"DEBUG: prepared point id: {point_id} (page {page_num_1based})")

        # upsert per batch
        if len(points_batch) >= BATCH_SIZE:
            print("DEBUG: upserting batch, sample ids:", [p.id for p in points_batch][:10])
            print("DEBUG: id types:", [type(p.id) for p in points_batch][:10])
            try:
                client.upsert(collection_name=collection_name, points=points_batch)
            except Exception:
                traceback.print_exc()
                print("DEBUG: Failed to upsert. Batch ids:", [p.id for p in points_batch])
                raise
            points_batch = []

    # final batch
    if points_batch:
        print("DEBUG: upserting final batch, sample ids:", [p.id for p in points_batch][:10])
        try:
            client.upsert(collection_name=collection_name, points=points_batch)
        except Exception:
            traceback.print_exc()
            print("DEBUG: Failed to upsert final batch. Batch ids:", [p.id for p in points_batch])
            raise

    print("Terminé. Pages insérées dans Qdrant.")

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Indexer pages PDF -> embeddings CLIP -> Qdrant (IDs UUID)")
    parser.add_argument("--pdf", required=True, help="Chemin vers le fichier PDF local")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333", help="URL Qdrant")
    parser.add_argument("--api-key", default=None, help="API key Qdrant (si configurée)")
    parser.add_argument("--collection", default="pdf_pages_images", help="Nom collection Qdrant")
    parser.add_argument("--out-images", default="./pdf_pages_images", help="Dossier pour sauvegarder les images de pages (optionnel)")
    parser.add_argument("--max-pages", type=int, default=None, help="Limiter le nombre de pages pour tests")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Nom du modèle CLIP (transformers) à utiliser")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"Fichier PDF introuvable : {pdf_path}")

    process_pdf_to_qdrant(pdf_path, args.qdrant, args.api_key, args.collection, Path(args.out_images), args.max_pages, model_name=args.model)

if __name__ == "__main__":
    main()
