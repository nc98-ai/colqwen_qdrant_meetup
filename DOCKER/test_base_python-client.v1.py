# test_python-client.v1.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

COLLECTION_NAME = "ma_collection2"
qdrant_apikey   = "ma_cle_secrete"

# connexion au service Qdrant
client = QdrantClient(url="http://127.0.0.1:6333", api_key=qdrant_apikey)  # api_key optionnel


# vérifier la liste des collections
print(client.get_collections())

exists = client.collection_exists(collection_name=COLLECTION_NAME)


if exists:
    print(f"--- La collection '{COLLECTION_NAME}' existe déjà.\n")
else:
    # creation collection
    print(f"--- La collection '{COLLECTION_NAME}' n'existe pas. Creation...\n")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    print(f"--- La collection '{COLLECTION_NAME}' a été créée avec succès.\n")



# affiche les propriétés de la collection
from qdrant_client import QdrantClient
import json

client = QdrantClient(url="http://127.0.0.1:6333", api_key=qdrant_apikey)
info = client.get_collection(COLLECTION_NAME)  # retourne les propriétés de la collection
# selon la version, 'info' peut être un dict ou un objet ; normaliser en dict :
try:
    print(f"--- Propriétés de la collection '{COLLECTION_NAME}':\n")
    printable = info.dict()
except AttributeError:
    printable = info
print(json.dumps(printable, indent=2, ensure_ascii=False))



#suppression collection
from qdrant_client import QdrantClient
print(f"--- La collection '{COLLECTION_NAME}' va être supprimée.\n")
client = QdrantClient(url="http://127.0.0.1:6333", api_key=qdrant_apikey)
client.delete_collection(collection_name=COLLECTION_NAME)
# vérifier l'absence
exists = client.collection_exists(collection_name=COLLECTION_NAME)
print("Existe encore :", exists)  # False attendu après suppression