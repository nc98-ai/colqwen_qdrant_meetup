import logging
import requests
import sys
import torch
from qdrant_client import QdrantClient
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers import BitsAndBytesConfig

# --- Config ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ColQwenSearch")

QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = "ma_cle_secrete"
COLLECTION_NAME = "rag_colqwen2"
MODEL_NAME = "vidore/colqwen2-v0.1"

# Ollama Vision
OLLAMA_API = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2-vision" # Assurez-vous de l'avoir pullé
OLLAMA_MODEL = "qwen3-vl:8b-instruct"

def get_model_and_processor():
    logger.info(f"Chargement du xmodèle {MODEL_NAME} (Query Mode)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = ColQwen2.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    processor = ColQwen2Processor.from_pretrained(MODEL_NAME)
    return model, processor

def search_qdrant(query_embedding, client, limit=1):
    """Recherche avec vecteurs multiples."""
    # Note: query_embedding est une liste de listes [[...], [...]] (un vecteur par token)
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding, # Qdrant gère nativement le multivector query ici
        limit=limit
    )
    return results.points

def query_ollama_vision(question, image_b64):
    """Envoie l'image trouvée et la question au VLM."""
    
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"Analyse cette image de document et réponds à la question : {question}",
                "images": [image_b64]
            }
        ],
        "stream": False,
        "options": {"temperature": 0.1}
    }

    try:
        logger.info(f"Envoi au LLM Vision ({OLLAMA_MODEL})...")
        resp = requests.post(OLLAMA_API, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        return f"Erreur LLM: {e}"

def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Quelle est la différence de medal rate entre gpt-oss 20b et 120b ?"

    logger.info(f"Question : {question}")

    # 1. Init
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    model, processor = get_model_and_processor()

    # 2. Vectorisation de la Question (ColQwen)
    # ColQwen traite la question comme une séquence de tokens, produisant plusieurs vecteurs
    batch_queries = processor.process_queries([question]).to(model.device)
    
    with torch.no_grad():
        # Shape: [1, N_Tokens, 128]
        query_embeddings = model(**batch_queries)
    
    # Conversion pour Qdrant
    q_vec = query_embeddings[0].cpu().float().numpy().tolist()

    # 3. Recherche
    logger.info("Recherche MaxSim dans Qdrant...")
    results = search_qdrant(q_vec, client, limit=30)

    if not results:
        print("Aucun résultat trouvé.")
        return

    best_hit = results[0]
    logger.info(f"Meilleur résultat : Page {best_hit.payload['page']} (Score: {best_hit.score})")
    
    image_b64 = best_hit.payload['image_base64']

    # 4. Génération
    answer = query_ollama_vision(question, image_b64)
    
    print("\n" + "="*40)
    print("RÉPONSE DU VLM :")
    print("="*40)
    print(answer)
    print("="*40)

if __name__ == "__main__":
    main()