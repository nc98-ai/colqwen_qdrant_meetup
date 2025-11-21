# pip install qdrant-client pdf2image pymupdf torch transformers accelerate bitsandbytes colpali-engine peft
import logging
import torch
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import uuid
from torch.utils.data import DataLoader
from tqdm import tqdm

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, MultiVectorConfig, MultiVectorComparator

from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers import BitsAndBytesConfig

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ColQwenIngest")

PDF_PATH = "DOCKER/Conditions Spécifiques du contrat d'abonnement  au service Trunk Sip_tagged.pdf"
PDF_PATH = "DOCKER/2511.15593v1.pdf"
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = "ma_cle_secrete"
COLLECTION_NAME = "rag_colqwen2"
MODEL_NAME = "vidore/colqwen2-v0.1"

def get_model_and_processor():
    """Charge ColQwen2 en 4-bits pour économiser la VRAM."""
    logger.info(f"Chargement de {MODEL_NAME} en 4-bit...")
    
    # Config 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = ColQwen2.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" # Force le GPU
    )
    
    processor = ColQwen2Processor.from_pretrained(MODEL_NAME)
    return model, processor

def init_qdrant(client: QdrantClient):
    """Init Qdrant avec configuration Multi-Vecteur (MaxSim)."""
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    
    logger.info("Création de la collection Qdrant Multi-Vecteur...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=128,  # ColQwen/ColPali projette toujours vers 128 dimensions
            distance=Distance.COSINE,
            # C'est ici que la magie opère pour ColBERT/ColPali
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            )
        )
    )

def pdf_to_images(path):
    doc = fitz.open(path)
    images = []
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200) # Bonne résolution pour ColQwen
        img_data = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_data))
        
        # On encode aussi en B64 pour le stockage (pour le VLM plus tard)
        img_b64 = base64.b64encode(img_data).decode('utf-8')
        
        images.append({
            "image": pil_image, 
            "page": page_num + 1, 
            "b64": img_b64
        })
    return images

def main():
    # 1. Chargement Modèle
    model, processor = get_model_and_processor()
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # 2. Init DB
    init_qdrant(client)
    
    # 3. Conversion PDF -> Images
    logger.info("Conversion du PDF en images...")
    pages_data = pdf_to_images(PDF_PATH)
    
    # 4. Ingestion
    logger.info("Génération des embeddings (Vision) et insertion...")
    
    # ColQwen traite image par image (ou par petits batchs)
    for page in tqdm(pages_data):
        image = page["image"]
        
        # Préparation ColQwen
        batch_images = processor.process_images([image]).to(model.device)
        
        # Inférence (No grad pour économiser mémoire)
        with torch.no_grad():
            # Retourne un tenseur [Batch, N_Patches, 128]
            embeddings = model(**batch_images)
        
        # Conversion pour Qdrant (on prend le 1er élément du batch)
        # .float() et .cpu() sont importants pour la compatibilité JSON/Qdrant
        multivector = embeddings[0].cpu().float().numpy().tolist()
        
        # Insertion
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=multivector, # Ceci est une LISTE de vecteurs (ex: [[0.1...], [0.2...]])
                    payload={
                        "page": page["page"],
                        "image_base64": page["b64"],
                        "local_PDF_PATH": PDF_PATH
                    }
                )
            ]
        )
        
        #
        logger.info(f"Page {page['page']}/{len(pages_data)} de {PDF_PATH} insérée dans Qdrant...")
        print(f"Page {page['page']}/{len(pages_data)} de {PDF_PATH} insérée dans Qdrant...")
        
        
    logger.info("Ingestion terminée avec succès.")

if __name__ == "__main__":
    main()