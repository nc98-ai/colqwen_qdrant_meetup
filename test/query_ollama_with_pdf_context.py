import logging
import requests
import sys
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# --- Configuration du Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RAGQuery")

# --- Configuration ---
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = "ma_cle_secrete"
COLLECTION_NAME = "rag_multimodal"

# Configuration Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "qwen2.5:14b-instruct-q5_0"
OLLAMA_MODEL = "qwen3-vl:8b-instruct"

# Modèle d'embedding (DOIT être le même que pour l'ingestion)
EMBEDDING_MODEL_NAME = "clip-ViT-B-32"

def get_embedding(text, model):
    """Génère l'embedding de la requête utilisateur via CLIP."""
    return model.encode(text).tolist()

def search_qdrant(query_vector, client, limit=3):
    """Recherche les vecteurs les plus proches (Textes ou Images)."""
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )
        return results
    except Exception as e:
        logger.error(f"Erreur lors de la recherche Qdrant: {e}")
        return []

def query_ollama(prompt, context_text):
    """Envoie le prompt construit à Ollama."""
    
    # Construction du prompt système pour Qwen
    system_prompt = (
        "Tu es un expert chargé d'analyser des documents. "
        "Réponds à la question en utilisant UNIQUEMENT les informations contextuelles fournies ci-dessous. "
        "Si l'information n'est pas dans le contexte, dis que tu ne sais pas."
    )
    
    full_prompt = (
        f"{system_prompt}\n\n"
        f"--- CONTEXTE (Texte et descriptions d'images extraits du PDF) ---\n"
        f"{context_text}\n"
        f"---------------------------------------------------------------\n\n"
        f"QUESTION: {prompt}\n"
        f"REPONSE:"
    )
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1  # Température basse pour être factuel
        }
    }
    
    try:
        logger.info(f"Envoi de la requête au LLM ({OLLAMA_MODEL})...")
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur de communication avec Ollama: {e}")
        return "Erreur : Impossible de contacter le modèle LLM."

def main():
    # Question par défaut ou via argument ligne de commande
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])
    else:
        # --- MODIFIEZ VOTRE QUESTION ICI ---
        user_question = "Quelles sont les conditions de résiliation ?"
    
    print(f"\n--- Question : {user_question} ---\n")

    # 1. Initialisation
    try:
        logger.info("Connexion à Qdrant...")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        logger.info(f"Chargement du modèle {EMBEDDING_MODEL_NAME}...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        logger.error(f"Erreur d'initialisation : {e}")
        return

    # 2. Embedding de la question
    query_vector = get_embedding(user_question, model)

    # 3. Recherche Multimodale dans Qdrant
    logger.info("Recherche dans la base vectorielle...")
    search_results = search_qdrant(query_vector, client, limit=3)

    if not search_results:
        logger.warning("Aucun résultat trouvé dans Qdrant.")
        return

    # 4. Construction du contexte
    context_parts = []
    for i, res in enumerate(search_results):
        payload = res.payload
        cw_type = payload.get("type", "unknown")
        page = payload.get("page", "?")
        content = payload.get("content", "").replace("\n", " ")
        
        logger.info(f"Résultat {i+1} (Score: {res.score:.3f}) - Page {page} [{cw_type}]")
        
        if cw_type == 'image':
            # Si on trouve une image, on le dit au LLM
            context_parts.append(f"[IMAGE Page {page}]: Une image pertinente a été trouvée ici.")
        else:
            context_parts.append(f"[TEXTE Page {page}]: {content}")

    full_context = "\n\n".join(context_parts)

    # 5. Appel au LLM
    logger.info("Génération de la réponse...")
    answer = query_ollama(user_question, full_context)
    
    print("\n" + "="*30)
    print("RÉPONSE DU LLM :")
    print("="*30)
    print(answer)
    print("="*30 + "\n")

if __name__ == "__main__":
    main()