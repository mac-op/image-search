import hashlib
import os
import pickle
from typing import Optional

import modal
import torch
import torch.nn.functional as F

# Modal app and image setup

image = modal.Image.from_dockerfile("./docker/dev.dockerfile")

app = modal.App("image-search")

# Modal volumes for model caching
models_volume = modal.Volume.from_name("image-search-models", create_if_missing=True)

# Redis client will be initialized on container startup
redis_client: Optional[any] = None


@app.cls(
    image=image,
    gpu=modal.gpu.T4(1),
    cpu=1.,
    # env=dict(
    #     PINECONE_API_KEY=modal.Secret.from_name("pinecone-secret"),
    #     PINECONE_INDEX_NAME="im-search",
    #     REDIS_URL=modal.Secret.from_name("redis-secret"),
    #     USE_GCP="0",
    # ),
    secrets=[
        modal.Secret.from_name("pinecone-secret"),
        modal.Secret.from_name("redis-secret"),
        modal.Secret.from_name("gcp-secret"),  # Contains GCP_SERVICE_ACCOUNT_JSON
    ],
    volumes={"/mnt": models_volume},
    region="us-central1"
)
class ImageSearchService:
    @modal.enter()
    def setup(self):
        """Initialize models and connections on container startup"""
        global redis_client

        # Import here to avoid issues with cold starts
        import json
        import redis.asyncio as redis
        from pinecone.grpc import PineconeGRPC as Pinecone
        from models import ModelConfig
        from cloud_storage import get_storage_client
        from misc.prepare_env import write_to_mount

        # Setup GCP authentication
        # Modal secrets expose environment variables, so we write the service account JSON to a file
        gcp_creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
        if gcp_creds_json:
            gcp_creds_path = "/tmp/gcp_credentials.json"
            with open(gcp_creds_path, "w") as f:
                # Handle both raw JSON string and base64 encoded
                try:
                    creds_data = json.loads(gcp_creds_json)
                    json.dump(creds_data, f)
                except json.JSONDecodeError:
                    # If it's base64 encoded
                    import base64
                    decoded = base64.b64decode(gcp_creds_json)
                    f.write(decoded.decode('utf-8'))

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_creds_path
            os.environ["USE_GCP"] = "1"

        # Prepare models (write to /mnt volume)
        write_to_mount()

        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = self.pc.Index(os.environ.get("PINECONE_INDEX_NAME", "im-search"))

        # Initialize models
        self.model_config = ModelConfig.for_inference()

        # Initialize cloud storage
        self.storage = get_storage_client()

        # Initialize Redis
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self.redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=False)
        redis_client = self.redis_client

    def rerank(self, prompt: str, keys: list[str], images: list[bytes]) -> list[tuple[str, float]]:
        """Rerank images using BLIP model"""
        images_tensors = [torch.frombuffer(img, dtype=torch.float16).view(3, 384, 384) for img in images]
        images_batch = torch.stack(images_tensors, dim=0).cuda()

        inputs = self.model_config.blip_preprocessor(images=images_batch, text=[prompt], return_tensors="pt")
        pixel_values = inputs.pixel_values.to("cuda").to(torch.float16)
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")

        with torch.no_grad():
            itm, cosine = self.model_config.blip_model(pixel_values, input_ids, attention_mask)
        scores = itm * 0.7 + cosine * 0.3
        return sorted(zip(keys, scores.cpu().tolist()), key=lambda x: x[1], reverse=True)

    async def get_cached_images(self, keys: list[str]) -> dict[str, bytes | None]:
        """Get images from cache, returns dict with key -> image bytes or None if not cached"""
        if not self.redis_client:
            return {key: None for key in keys}

        cache_keys = [f"img:{key}" for key in keys]
        cached = await self.redis_client.mget(cache_keys)
        return {key: (pickle.loads(data) if data else None) for key, data in zip(keys, cached)}

    async def cache_images(self, images_dict: dict[str, bytes], ttl: int = 3600):
        """Cache images with TTL (default 1 hour)"""
        if not self.redis_client or not images_dict:
            return

        pipeline = self.redis_client.pipeline()
        for key, img_data in images_dict.items():
            cache_key = f"img:{key}"
            await pipeline.setex(cache_key, ttl, pickle.dumps(img_data))
        await pipeline.execute()

    @modal.method()
    async def search(self, query: str) -> dict:
        """
        Search for images using text query.
        Returns results with scores and caching info.
        """
        cache_key = ""
        if self.redis_client:
            # Check cache for query results
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            cache_key = f"search:{query_hash}"

            cached_result = await self.redis_client.get(cache_key)
            if cached_result:
                return {"results": pickle.loads(cached_result), "cached": True}

        # Encode query using SigLIP
        tokens = self.model_config.siglip_tokenizer([query]).cuda()

        with torch.no_grad():
            text_embedding = self.model_config.siglip_text(tokens)
            text_embedding = F.normalize(text_embedding, dim=-1)
            text_embedding = text_embedding.cpu().numpy()[0]

        # Query Pinecone
        results = self.index.query(
            vector=text_embedding.tolist(),
            top_k=32,
            include_metadata=True
        )

        keys = [match.id for match in results.matches]

        # Try to get images from cache first
        cached_images = await self.get_cached_images(keys)
        missing_keys = [k for k, v in cached_images.items() if v is None]

        # Fetch missing images from storage
        if missing_keys:
            fetched_images = await self.storage.bulk_get(missing_keys)
            # Cache newly fetched images
            await self.cache_images({k: img for k, img in zip(missing_keys, fetched_images)})
            # Merge cached and fetched
            all_images = [cached_images[k] if cached_images[k] is not None
                          else fetched_images[missing_keys.index(k)] for k in keys]
        else:
            all_images = [cached_images[k] for k in keys]

        # Rerank using BLIP
        result = self.rerank(query, keys, all_images)

        # Cache search results for 5 minutes
        if self.redis_client:
            await self.redis_client.setex(cache_key, 300, pickle.dumps(result))

        return {"results": result, "cached": False}

    @modal.method()
    async def health_check(self) -> dict:
        """Health check endpoint"""
        return {"status": "ok"}


# Web endpoints
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("pinecone-secret")],
    region="us-central1"
)
@modal.fastapi_endpoint(method="POST", label="search")
async def search_endpoint(data: dict):
    """
    Primary endpoint for image search.
    Expects JSON: {"query": "description of the image"}
    Returns: {"results": [[id, score], ...], "cached": bool}
    """
    query = data.get("query", "")
    if not query:
        return {"error": "No query provided"}, 400

    service = ImageSearchService()
    return await service.search.remote(query)


@app.function(image=image)
@modal.fastapi_endpoint(method="GET", label="health")
async def health_endpoint():
    """Health check endpoint"""
    service = ImageSearchService()
    return await service.health_check.remote()
