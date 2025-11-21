import logging
import os
import io
import fitz  # PyMuPDF
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# --- Configuration du Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("IngestionScript")

# --- Configuration ---
PDF_PATH = "DOCKER/Conditions Spécifiques du contrat d'abonnement  au service Trunk Sip_tagged.pdf"
PDF_PATH = "DOCKER/2511.15593v1.pdf"
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = "ma_cle_secrete"
COLLECTION_NAME = "rag_multimodal"
EMBEDDING_MODEL_NAME = "clip-ViT-B-32"

def init_qdrant(client: QdrantClient, vector_size: int):
    """Initialise la collection Qdrant."""
    try:
        if client.collection_exists(COLLECTION_NAME):
            logger.info(f"La collection '{COLLECTION_NAME}' existe déjà. Suppression pour recréation...")
            client.delete_collection(COLLECTION_NAME)
        
        logger.info(f"Création de la collection '{COLLECTION_NAME}' avec une taille de vecteur de {vector_size}...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Collection créée avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de Qdrant: {e}")
        raise

def process_pdf_and_index(pdf_path: str, client: QdrantClient, model: SentenceTransformer):
    """Lit le PDF, extrait texte et images, vectorise et indexe dans Qdrant."""
    
    if not os.path.exists(pdf_path):
        logger.error(f"Le fichier PDF n'a pas été trouvé : {pdf_path}")
        return

    try:
        doc = fitz.open(pdf_path)
        logger.info(f"Ouverture du PDF : {pdf_path} ({len(doc)} pages)")
        
        points = []
        
        for page_num, page in enumerate(doc):
            logger.info(f"Traitement de la page {page_num + 1}...")
            
            # 1. Traitement du Texte
            text_content = page.get_text()
            if text_content.strip():
                # Vectorisation du texte
                text_embedding = model.encode(text_content).tolist()
                point_id = str(uuid.uuid4())
                
                points.append(PointStruct(
                    id=point_id,
                    vector=text_embedding,
                    payload={
                        "source": pdf_path,
                        "page": page_num + 1,
                        "type": "text",
                        "content": text_content
                    }
                ))

            # 2. Traitement des Images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Chargement de l'image pour PIL
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Vectorisation de l'image (Multimodal)
                    image_embedding = model.encode(image).tolist()
                    point_id_img = str(uuid.uuid4())
                    
                    points.append(PointStruct(
                        id=point_id_img,
                        vector=image_embedding,
                        payload={
                            "source": pdf_path,
                            "page": page_num + 1,
                            "type": "image",
                            "image_index": img_index,
                            "content": f"[IMAGE trouvée page {page_num + 1}]"
                        }
                    ))
                    logger.debug(f"Image {img_index} vectorisée (page {page_num + 1})")
                    
                except Exception as img_err:
                    logger.warning(f"Impossible de traiter une image page {page_num + 1}: {img_err}")

        # Insertion en batch
        if points:
            logger.info(f"Insertion de {len(points)} points dans Qdrant...")
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            logger.info("Insertion terminée.")
        else:
            logger.warning("Aucun contenu extractible trouvé.")

    except Exception as e:
        logger.error(f"Erreur critique lors du traitement du PDF: {e}")
    finally:
        if 'doc' in locals():
            doc.close()

def main():
    logger.info("Démarrage du script d'ingestion...")
    
    # Chargement du modèle CLIP (Multimodal)
    logger.info(f"Chargement du modèle {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # CORRECTION : Calculer la taille du vecteur dynamiquement
    # Car model.get_sentence_embedding_dimension() peut renvoyer None pour CLIP
    logger.info("Calcul de la dimension du vecteur...")
    dummy_vector = model.encode("test")
    vector_size = len(dummy_vector)
    logger.info(f"Dimension détectée : {vector_size}")
    
    # Connexion Qdrant
    # Note: API KEY utilisée sur HTTP (non HTTPS) génère un warning, c'est normal en local
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Init DB
    init_qdrant(client, vector_size)
    
    # Processing
    process_pdf_and_index(PDF_PATH, client, model)
    
    logger.info("Fin du script d'ingestion.")

if __name__ == "__main__":
    main()