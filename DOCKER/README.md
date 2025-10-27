# telecharge image
docker pull qdrant/qdrant:v1.13.3-gpu-nvidia

# démarrer un conteneur avec prise en charge du GPU  basée sur cette image
méthode 1: docker compose
```yaml
# docker-compose-qdrant-nvidia.yml
version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant:v1.13.3-gpu-nvidia
    container_name: qdrant-nvidia
    restart: unless-stopped
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant-data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__API_KEY=ma_cle_secrete
    deploy: {}   # conservé vide si vous n'êtes pas en swarm
    # Alternative plus simple et souvent supportée par docker-compose:
    gpus: all
```
```bash
# démarrer
docker compose -f docker-compose-qdrant-nvidia.yml up -d
# tester le GPU
dke qdrant-nvidia nvidia-smi
#logs
docker logs -f qdrant-nvidia
# health
docker inspect --format='{{json .State.Health}}' qdrant-nvidia
```



# consulter la version déployée
curl -v http://127.0.0.1:6333


# console administrateur
http://127.0.0.1:6333/dashboard


# Lister les collections
curl -sS -X GET "http://127.0.0.1:6333/collections" -H "api-key: ma_cle_secrete"

# creer une collection
- methode 1: via API
```bash
curl -sS -X PUT "http://127.0.0.1:6333/collections/ma_collection" \
  -H "Content-Type: application/json" \
  -H "api-key: ma_cle_secrete" \
  -d '{
    "vectors": {
      "size": 1536,
      "distance": "Cosine"
    }
  }'
```

- methode 2: via python avec qdrant-client  
```bash
pip install qdrant-client

```

```python
# test_python-client.v1.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

#connexion
client = QdrantClient(url="http://127.0.0.1:6333", api_key="ma_cle_secrete")  # api_key optionnel


#creation collection
client.create_collection(
    collection_name="ma_collection2",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# liste des collections
print(client.get_collections())
```

# affiche les proprités d'une collection
- methode 1: curl
```bash
curl -sS -X GET "http://127.0.0.1:6333/collections/ma_collection" \
  -H "api-key: ma_cle_secrete" \
  | jq .
```

- methode 2: via python avec qdrant-client  


