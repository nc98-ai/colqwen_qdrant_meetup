import logging
import requests
import sys
import torch
import re  # Pour détecter "page X"
from qdrant_client import QdrantClient, models
from transformers import BitsAndBytesConfig

# --- IMPORT ADAPTÉ POUR COLQWEN 2.5 ---
from colpali_engine.models import ColQwen2_5, ColQwen2Processor

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ColQwenSearch")

QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = "ma_cle_secrete"
COLLECTION_NAME = "rag_colqwen2.5"

MODEL_NAME = "vidore/colqwen2.5-v0.2"
OLLAMA_API = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3-vl:8b-instruct"

def get_model_and_processor():
    logger.info(f"Chargement du modèle {MODEL_NAME}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = ColQwen2_5.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    processor = ColQwen2Processor.from_pretrained(MODEL_NAME)
    return model, processor

def search_qdrant(query_embeddings, client, filter_condition=None, limit=3):
    """
    Recherche avec vecteurs multiples (MaxSim) ET filtre optionnel.
    """
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embeddings, 
        query_filter=filter_condition, # Application du filtre ici
        limit=limit
    )
    return results.points

def query_ollama_vision(question, search_results):
    """Envoie les images trouvées (Top-K) au VLM."""
    
    images_list = []
    context_desc = ""
    
    for i, res in enumerate(search_results):
        payload = res.payload
        page_num = payload.get('page', '?')
        score = res.score
        img_b64 = payload.get('image_base64')
        
        if img_b64:
            images_list.append(img_b64)
            context_desc += f"- Image {i+1} : Page {page_num} (Score de pertinence: {score:.2f})\n"

    if not images_list:
        return "Erreur : Aucune image trouvée."

    system_msg = (
        "Tu es un expert analyste. On te fournit une ou plusieurs images de pages de document. "
        "Identifie l'image la plus pertinente par rapport à la question et réponds en utilisant ses informations visuelles."
    )

    user_msg = f"""Voici {len(images_list)} image(s) extraite(s) du document :
    {context_desc}
    
    QUESTION UTILISATEUR : "{question}"
    
    Réponds précisément à la question en citant la page utilisée."""
    
    # Construction du message pour Ollama
    # Note: L'API chat d'Ollama supporte une liste d'images dans un seul message
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": user_msg,
                "images": images_list
            }
        ],
        "stream": False,
        "options": {"temperature": 0.1}
    }

    try:
        logger.info(f"Envoi de {len(images_list)} image(s) au LLM ({OLLAMA_MODEL})...")
        resp = requests.post(OLLAMA_API, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        return f"Erreur LLM: {e}"

def extract_page_filter(question):
    """Détecte si l'utilisateur demande une page spécifique."""
    # Regex simple pour chercher "page X" ou "p. X"
    match = re.search(r"(?:page|p\.|p)\s*(\d+)", question.lower())
    if match:
        page_num = int(match.group(1))
        logger.info(f"🔍 Filtre détecté : Restriction à la page {page_num}")
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="page",
                    match=models.MatchValue(value=page_num)
                )
            ]
        )
    return None

def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "dans la page 11, quel est environ le pourcentage de Medal Rate de devstral?"

    logger.info(f"Question : {question}")

    # 1. Init
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    model, processor = get_model_and_processor()

    # 2. Embedding Question
    batch_queries = processor.process_queries([question]).to(model.device)
    with torch.no_grad():
        query_embeddings_tensor = model(**batch_queries)
    q_vec = query_embeddings_tensor[0].cpu().float().numpy().tolist()

    # 3. Préparation Filtre
    qdrant_filter = extract_page_filter(question)

    # 4. Recherche (Top 3 par défaut pour augmenter les chances si pas de filtre)
    # Si un filtre est appliqué, Qdrant ne cherchera QUE dans cette page.
    logger.info("Recherche Qdrant...")
    results = search_qdrant(q_vec, client, filter_condition=qdrant_filter, limit=3)

    if not results:
        print("❌ Aucun résultat trouvé (vérifiez si le numéro de page existe).")
        return

    # Log des pages trouvées
    found_pages = [str(r.payload.get('page')) for r in results]
    logger.info(f"Pages récupérées : {', '.join(found_pages)}")

    # 5. Génération
    answer = query_ollama_vision(question, results)
    
    print("\n" + "="*50)
    print(f"🤖 RÉPONSE DU VLM ({OLLAMA_MODEL}) :")
    print("="*50)
    print(answer)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()