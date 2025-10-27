#!/usr/bin/env python3
"""
script_pdf_to_qdrant_fixed.py

Extrait les pages d'un PDF -> image -> embedding (ResNet par défaut) -> upsert dans Qdrant.
Corrige le problème d'ID en envoyant des UUID (string) et ajoute du debug avant upsert.
Usage:
python script_pdf_to_qdrant_fixed.py --pdf trunksip.pdf --qdrant http://127.0.0.1:6333 --api-key ma_cle_secrete
"""

import sys
import os
import argparse
import uuid
from pathlib import Path
from tqdm import tqdm
import traceback

import fitz  # PyMuPDF
from PIL import Image
import numpy as np

# Torch & torchvision (ResNet léger par défaut)
import torch
import torchvision.transforms as T
from torchvision import models

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

# OCR & language detection
import pytesseract
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # pour stabilité

# Debug info : quel fichier est exécuté
print("DEBUG: Running script file:", __file__, " (python executable:", sys.executable, ")")

# ---------- CONFIG ----------
BATCH_SIZE = 64        # upsert par lot
RENDER_DPI = 150       # qualité rendu page -> image
VECTOR_NAME = "page_image"
DISTANCE = Distance.COSINE
DEFAULT_VECTOR_DIM = 512  # resnet18 => 512

# ---------- Helpers ----------
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

# ---------- ResNet Embedder ----------
class ResNetEmbedder:
    def __init__(self, device=None, model_name='resnet18', pretrained=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == 'resnet18':
            base = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            base = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError("model_name non supporté; choisissez 'resnet18' ou 'resnet50'")

        modules = list(base.children())[:-1]
        self.model = torch.nn.Sequential(*modules).to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def embed_image(self, pil_image: Image.Image):
        img = pil_image.convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model(x)
        feats = feats.reshape(feats.size(0), -1).cpu().numpy()[0]
        norm = np.linalg.norm(feats)
        if norm > 0:
            feats = feats / norm
        return feats.tolist()

# ---------- Qdrant helpers ----------
def ensure_collection(client: QdrantClient, collection_name: str, vector_name: str, vector_dim: int):
    try:
        exists = client.collection_exists(collection_name=collection_name)
    except Exception as e:
        raise RuntimeError(f"Erreur connexion Qdrant: {e}")
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={vector_name: VectorParams(size=vector_dim, distance=DISTANCE)}
        )
        print(f"Collection '{collection_name}' créée (vector {vector_name}, dim={vector_dim}).")
    else:
        print(f"Collection '{collection_name}' existe déjà.")

# ---------- Main processing ----------
def process_pdf_to_qdrant(pdf_path: Path, qdrant_url: str, api_key: str, collection_name: str,
                          out_image_dir: Path = None, max_pages: int = None, model_name='resnet18'):
    # client Qdrant
    client = QdrantClient(url=qdrant_url, api_key=api_key)
    # embedder
    embedder = ResNetEmbedder(model_name=model_name)
    vector_dim = embedder.feature_dim

    ensure_collection(client, collection_name, VECTOR_NAME, vector_dim)

    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count if max_pages is None else min(doc.page_count, max_pages)

    if out_image_dir:
        out_image_dir.mkdir(parents=True, exist_ok=True)

    points_batch = []

    for page_number in tqdm(range(total_pages), desc="Pages"):
        page = doc.load_page(page_number)
        page_num_1based = page_number + 1

        text = extract_text_from_page(page)
        lang = detect_language(text) if text else None

        pil_img = render_page_to_pil(page)

        if not text:
            ocr_text = ocr_image(pil_img, lang_hint=lang)
            text = ocr_text
            lang = detect_language(text) or lang

        saved_image_path = None
        if out_image_dir:
            img_name = f"{pdf_path.stem}_page_{page_num_1based}.png"
            saved_image_path = str(out_image_dir / img_name)
            try:
                pil_img.save(saved_image_path)
            except Exception as e:
                print(f"Warning: impossible de sauvegarder l'image {saved_image_path}: {e}")
                saved_image_path = None

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

        # --- IMPORTANT: generate UUID string for id (accepted by Qdrant) ---
        point_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"{pdf_path.resolve()}#p{page_num_1based}")
        point_id = str(point_uuid)   # <-- string UUID

        point = PointStruct(id=point_id, vector={VECTOR_NAME: vec}, payload=payload)
        points_batch.append(point)

        # debug: show prepared id
        print(f"DEBUG: prepared point id: {point_id} (type: {type(point_id)}) for page {page_num_1based}")

        # upsert by batch
        if len(points_batch) >= BATCH_SIZE:
            # debug: show sample ids we are about to send
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
    parser = argparse.ArgumentParser(description="Indexer pages PDF -> embeddings ResNet -> Qdrant (IDs UUID)")
    parser.add_argument("--pdf", required=True, help="Chemin vers le fichier PDF local")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333", help="URL Qdrant")
    parser.add_argument("--api-key", default=None, help="API key Qdrant (si configurée)")
    parser.add_argument("--collection", default="pdf_pages_images", help="Nom collection Qdrant")
    parser.add_argument("--out-images", default="./pdf_pages_images", help="Dossier pour sauvegarder les images de pages (optionnel)")
    parser.add_argument("--max-pages", type=int, default=None, help="Limiter le nombre de pages pour tests")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"], help="Modèle torchvision à utiliser pour embeddings")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"Fichier PDF introuvable : {pdf_path}")

    process_pdf_to_qdrant(pdf_path, args.qdrant, args.api_key, args.collection, Path(args.out_images), args.max_pages, model_name=args.model)

if __name__ == "__main__":
    main()
