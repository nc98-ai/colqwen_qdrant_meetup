import logging
import torch
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import uuid
from tqdm import tqdm

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, MultiVectorConfig, MultiVectorComparator

# --- CORRECTION DES IMPORTS ---
# On importe le MODELE spécifique version 2.5
# Mais on garde le PROCESSOR de la version 2 (car il est compatible et la v2.5 n'a pas sa propre classe processor)
from colpali_engine.models import ColQwen2_5, ColQwen2Processor
from transformers import BitsAndBytesConfig

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ColQwenIngest")

PDF_PATH = "DOCKER/2511.15593v1.pdf" # Votre PDF
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = "ma_cle_secrete"
COLLECTION_NAME = "rag_colqwen2.5"

# Le modèle Vidore 2.5 officiel
MODEL_NAME = "vidore/colqwen2.5-v0.2"

def get_model_and_processor():
    """Charge ColQwen2.5 en 4-bits."""
    logger.info(f"Chargement de {MODEL_NAME} avec la classe ColQwen2_5...")
    
    # Config 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- CORRECTION MAJEURE ICI ---
    # Utilisation de ColQwen2_5 (obligatoire pour l'architecture Qwen2.5-VL)
    model = ColQwen2_5.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    
    # Utilisation de ColQwen2Processor (le processeur est commun/compatible)
    processor = ColQwen2Processor.from_pretrained(MODEL_NAME)
    
    return model, processor

def init_qdrant(client: QdrantClient):
    if client.collection_exists(COLLECTION_NAME):
        logger.info(f"Suppression de l'ancienne collection {COLLECTION_NAME}...")
        client.delete_collection(COLLECTION_NAME)
    
    logger.info("Création de la collection Qdrant Multi-Vecteur...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=128,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            )
        )
    )

def pdf_to_images(path):
    try:
        doc = fitz.open(path)
    except Exception as e:
        logger.error(f"Erreur ouverture PDF: {e}")
        return []
        
    images = []
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        img_data = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_data))
        img_b64 = base64.b64encode(img_data).decode('utf-8')
        
        images.append({"image": pil_image, "page": page_num + 1, "b64": img_b64})
    return images

def main():
    try:
        model, processor = get_model_and_processor()
    except ImportError:
        logger.error("ERREUR D'IMPORT : La classe ColQwen2_5 n'est pas trouvée.")
        logger.error("Veuillez mettre à jour colpali-engine depuis GitHub :")
        logger.error("pip install --upgrade git+https://github.com/illuin-tech/colpali.git")
        return

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    init_qdrant(client)
    
    pages_data = pdf_to_images(PDF_PATH)
    if not pages_data: return

    logger.info(f"Ingestion de {len(pages_data)} pages...")
    
    for page in tqdm(pages_data):
        image = page["image"]
        
        # Traitement image
        batch_images = processor.process_images([image]).to(model.device)
        
        with torch.no_grad():
            embeddings = model(**batch_images)
        
        multivector = embeddings[0].cpu().float().numpy().tolist()
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=multivector,
                    payload={
                        "page": page["page"],
                        "image_base64": page["b64"],
                        "local_PDF_PATH": PDF_PATH,
                        "model": MODEL_NAME
                    }
                )
            ]
        )

    logger.info("Terminé.")

if __name__ == "__main__":
    main()