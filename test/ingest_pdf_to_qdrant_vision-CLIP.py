import logging
import os
import io
import base64
import fitz  # PyMuPDF
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import torch

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IngestBase64")

# --- Config ---
PDF_PATH = "DOCKER/Conditions Spécifiques du contrat d'abonnement  au service Trunk Sip_tagged.pdf"
PDF_PATH = "DOCKER/2511.15593v1.pdf"
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = "ma_cle_secrete"
COLLECTION_NAME = "rag_multimodal_vision" # Changement de nom pour éviter les conflits
EMBEDDING_MODEL = "clip-ViT-B-32"
EMBEDDING_MODEL = "google/siglip-so400m-patch14-384" #marche po, necessite pip install -U sentence-transformers sentencepiece timm
EMBEDDING_MODEL = "openai/clip-vit-large-patch14-336" #marche po
EMBEDDING_MODEL ="clip-ViT-L-14"


def init_qdrant(client, vector_size):
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

def process_pdf(pdf_path, client, model):
    doc = fitz.open(pdf_path)
    points = []
    
    for page_num, page in enumerate(doc):
        logger.info(f"Page {page_num + 1}...")
        
        # 1. TEXTE
        text = page.get_text()
        if text.strip():
            vec = model.encode(text).tolist()
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"type": "text", "page": page_num+1, "content": text}
            ))

        # 2. IMAGES (Correction majeure ici)
        img_list = page.get_images(full=True)
        for idx, img_info in enumerate(img_list):
            xref = img_info[0]
            base_img = doc.extract_image(xref)
            img_bytes = base_img["image"]
            
            # Conversion Base64 pour le stockage
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            try:
                # Vectorisation pour la recherche (CLIP)
                pil_img = Image.open(io.BytesIO(img_bytes))
                vec = model.encode(pil_img).tolist()
                
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "type": "image", 
                        "page": page_num+1, 
                        "image_base64": img_b64, # ON STOCKE L'IMAGE BRUTE
                        "content": "Contenu visuel (graphique/image)"
                    }
                ))
                logger.info(f"Image {idx} page {page_num+1} stockée.")
            except Exception as e:
                logger.warning(f"Erreur image: {e}")

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"{len(points)} chunks insérés.")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    
    # Calcul dimension dynamique
    dim = len(model.encode("test"))
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    init_qdrant(client, dim)
    process_pdf(PDF_PATH, client, model)

if __name__ == "__main__":
    main()