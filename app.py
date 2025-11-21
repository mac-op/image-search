import hashlib
import os
import pickle
from contextlib import asynccontextmanager

import redis.asyncio as redis
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.params import Depends
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pinecone.grpc import PineconeGRPC as Pinecone
from redis.asyncio import Redis

from cloud_storage import get_storage_client
from model_config import ModelConfig
from misc.prepare_env import write_to_mount

write_to_mount()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ.get("PINECONE_INDEX_NAME", "im-search"))
model_config = ModelConfig.for_inference()
storage = get_storage_client()

redis_client: Redis

@asynccontextmanager
async def lifespan(app_: FastAPI):
    global redis_client
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=False)
    await FastAPILimiter.init(redis_client)
    yield
    await redis_client.aclose()

app = FastAPI(
    title="Image Search",
    description="Search for images with a text prompt against a vector database .",
    version="0.1.0",
    lifespan=lifespan,
)


def rerank(prompt: str, keys: list[str], images: list[bytes]) -> list[tuple[str, float]]:
    images = [torch.frombuffer(img, dtype=torch.float16).view(3, 384, 384) for img in images]
    images = torch.stack(images, dim=0).cuda()

    inputs = model_config.blip_preprocessor(images=images, text=[prompt], return_tensors="pt")
    pixel_values = inputs.pixel_values.to("cuda").to(torch.float16)
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    with torch.no_grad():
        itm, cosine = model_config.blip_model(pixel_values, input_ids, attention_mask)
    scores = itm * 0.7 + cosine * 0.3
    return sorted(zip(keys, scores.cpu().tolist()), key=lambda x: x[1], reverse=True)

async def get_cached_images(keys: list[str]) -> dict[str, bytes | None]:
    """Get images from cache, returns dict with key -> image bytes or None if not cached"""
    if not redis_client:
        return {key: None for key in keys}

    cache_keys = [f"img:{key}" for key in keys]
    cached = await redis_client.mget(cache_keys)
    return {key: (pickle.loads(data) if data else None) for key, data in zip(keys, cached)}

async def cache_images(images_dict: dict[str, bytes], ttl: int = 3600):
    """Cache images with TTL (default 1 hour)"""
    if not redis_client or not images_dict:
        return

    pipeline = redis_client.pipeline()
    for key, img_data in images_dict.items():
        cache_key = f"img:{key}"
        await pipeline.setex(cache_key, ttl, pickle.dumps(img_data))
    await pipeline.execute()

@app.post("/search", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def im_search(data: dict):
    query = data.get("query", "")

    cache_key = ""
    if redis_client:
        # Check cache for query results
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        cache_key = f"search:{query_hash}"

        cached_result = await redis_client.get(cache_key)
        if cached_result:
            return {"results": pickle.loads(cached_result)}

    tokens = model_config.siglip_tokenizer([query]).cuda()

    with torch.no_grad():
        text_embedding = model_config.siglip_text_model(tokens)
        text_embedding = F.normalize(text_embedding, dim=-1)
        text_embedding = text_embedding.cpu().numpy()[0]

    results = index.query(
        vector=text_embedding.tolist(),
        top_k=32,
        include_metadata=True
    )

    if not results.matches:
        return {"results": []}

    keys = [match.id for match in results.matches]

    # Try to get images from cache first
    cached_images = await get_cached_images(keys)
    missing_keys = [k for k, v in cached_images.items() if v is None]

    # Fetch missing images from storage
    if missing_keys:
        fetched_images = await storage.bulk_get(missing_keys)
        # Cache newly fetched images
        await cache_images({k: img for k, img in zip(missing_keys, fetched_images)})
        # Merge cached and fetched
        all_images = [cached_images[k] if cached_images[k] is not None
                      else fetched_images[missing_keys.index(k)] for k in keys]
    else:
        all_images = [cached_images[k] for k in keys]

    result = rerank(query, keys, all_images)

    # Cache search results for 5 minutes
    if redis_client:
        await redis_client.setex(cache_key, 300, pickle.dumps(result))

    return {"results": result}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
