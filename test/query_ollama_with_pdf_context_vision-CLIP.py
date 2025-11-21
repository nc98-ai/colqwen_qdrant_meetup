import logging
import requests
import sys
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# --- Config ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAGVision")

QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = "ma_cle_secrete"
COLLECTION_NAME = "rag_multimodal_vision" # Même nom que le script 1

# IMPORTANT : Utiliser un modèle capable de voir (Vision-Language Model)
# Assurez-vous d'avoir fait: ollama pull llama3.2-vision
# OLLAMA_MODEL = "llama3.2-vision" 
OLLAMA_MODEL = "qwen3-vl:8b-instruct"
OLLAMA_API = "http://localhost:11434/api/chat"

EMBEDDING_MODEL = "clip-ViT-B-32"
EMBEDDING_MODEL = "google/siglip-so400m-patch14-384" #marche po, necessite pip install -U sentence-transformers sentencepiece timm
EMBEDDING_MODEL = "openai/clip-vit-large-patch14-336" #marche po
EMBEDDING_MODEL ="clip-ViT-L-14"

def query_ollama_vision(question, search_results):
    """Construit la requête multimodale pour Ollama."""
    
    messages = []
    
    # 1. On prépare le contexte textuel
    text_context = "Voici des extraits de texte du document :\n"
    images_b64 = []
    
    for res in search_results:
        payload = res.payload
        p_type = payload.get("type")
        
        if p_type == "text":
            text_context += f"- (Page {payload['page']}): {payload['content']}\n"
        
        elif p_type == "image":
            # Si c'est une image, on récupère le b64 stocké
            img_b64 = payload.get("image_base64")
            if img_b64:
                images_b64.append(img_b64)
                text_context += f"- (Page {payload['page']}): [IMAGE INCLUSE CI-JOINTE]\n"

    # 2. Construction du message utilisateur
    user_content = f"""Tu es un analyste expert. Utilise le texte et les images fournies pour répondre.
    
    CONTEXTE TEXTUEL:
    {text_context}
    
    QUESTION:
    {question}
    """

    # Format spécifique Ollama Chat pour les images
    # On attache toutes les images trouvées au message
    message_payload = {
        "role": "user",
        "content": user_content,
    }
    
    if images_b64:
        logger.info(f"{len(images_b64)} images injectées dans le prompt LLM.")
        message_payload["images"] = images_b64
    else:
        logger.warning("Aucune image trouvée dans les résultats de recherche.")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [message_payload],
        "stream": False,
        "options": {"temperature": 0.1}
    }

    try:
        logger.info(f"Envoi à Ollama ({OLLAMA_MODEL})...")
        resp = requests.post(OLLAMA_API, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        logger.error(f"Erreur Ollama: {e}")
        return "Erreur technique LLM"

def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "donne la difference entre le 'medal rate' de gpt-oss 20b et gpt-oss 120b"

    logger.info(f"Question: {question}")

    # Init
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Recherche Vectorielle (CLIP)
    # On augmente la limite pour espérer attraper l'image même si le score est moyen
    vector = model.encode(question).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=5
    )

    if not results:
        print("Rien trouvé dans Qdrant.")
        return

    # Génération Multimodale
    answer = query_ollama_vision(question, results)

    print("\n" + "="*40)
    print(f"RÉPONSE ({OLLAMA_MODEL}) :")
    print("="*40)
    print(answer)
    print("="*40)

if __name__ == "__main__":
    main()