# Image Search Backend

This repository contains the backend service for a semantic Reddit image search engine. Users can describe images in natural language and receive the most relevant matches from a vector database using multimodal AI models.

*Built with FastAPI, TensorRT, HuggingFace and Pinecone.*
## Overview

This is part of a Reddit image search engine. As part of building the engine, images are fetched, bulk processed by an 
embedding model, and stored in a vector database. Scheduled tasks are used to update the database with new images.

This service serves text-to-image search endpoint where users can query with descriptions like "a confused cat looking at a computer" and receive ranked image results. 
The pipeline combines fast vector similarity search with neural reranking for more accuracy.

## Architecture

### Core Components

1. **Text Encoding**: SigLIP text encoder converts user queries to embeddings
2. **Vector Search**: Pinecone vector database performs initial similarity matching  
3. **Image Retrieval**: Google Cloud Storage provides scalable image storage with Redis caching
4. **Neural Reranking**: BLIP model scores image-text pairs using:
   - Image-Text Matching (ITM) scores
   - Cosine similarity between embeddings
5. **Final Ranking**: Results ranked by `0.7 × ITM + 0.3 × cosine similarity`

### Deployment Options

The service supports multiple deployment strategies:

- FastAI app, to be deployed on Google Cloud Run or managed VPS of choice (`service/app.py`)
- Alternative serverless deployment with [Modal](https://modal.com) (`service/modal_app.py`)

[//]: # (- **Google Cloud Functions** &#40;`function/function.py`&#41;: Event-driven serverless execution)

### Technology Stack

- **Models**: SigLIP (text/image encoding), BLIP (multimodal reranking) provided by HuggingFace and OpenCLIP
- **Model Optimization**: TensorRT for accelerated inference
- **Vector Database**: Pinecone for similarity search
- **Storage**: Google Cloud Storage for image data
- **Caching**: Redis for query and image caching
- **API**: FastAPI with rate limiting and health checks, or Modal
- **Deployment**: Docker, Modal, Google Cloud Functions

## Prerequisites
- Python 3.12+
- CUDA-capable GPU (for local development, TensorRT is a required component for model compilation)
- Google Cloud Platform account
- Pinecone account
- Redis instance (for caching)

## Installation

1. See `docker/base.dockerfile` for required dependencies. Better yet, use the dev Docker image provided.

2. Set up environment variables:
```bash
export PINECONE_API_KEY=your_pinecone_api_key
export PINECONE_INDEX_NAME=image-search
export GOOGLE_CLOUD_PROJECT=your_project_id
export REDIS_URL=redis://localhost:6379  # Optional, for caching
```

## Deployment

### Option 1: Local FastAPI Service

```bash
# Start Redis (optional, for caching)
docker run -d -p 6379:6379 redis:alpine

# Run the service
uv run python -m uvicorn service.app:app --host 0.0.0.0 --port 8000
```

### Option 2: Modal Deployment

```bash
# Install and setup Modal
modal setup

# Deploy to Modal
modal deploy service/modal_app.py
```

### Option 3: Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Option 4: Google Cloud Functions

```bash
# Deploy function
gcloud functions deploy image-search \
  --source function/ \
  --entry-point main \
  --runtime python312 \
  --trigger-http \
  --allow-unauthenticated
```

## Usage

### API Endpoints

**POST** `/search`

Search for images matching a text description.

Request body:
```json
{
  "query": "a surprised pikachu face"
}
```

Response:
```json
{
  "results": [
    ["image_id_1", 0.95],
    ["image_id_2", 0.87],
    ["image_id_3", 0.82]
  ]
}
```
Image IDs are Reddit post IDs.\
This endpoint includes rate limiting for FastAPI to prevent abuse.

## Acknowledgements

- [SigLIP](https://arxiv.org/abs/2303.15343) by Google Research
- [BLIP](https://arxiv.org/abs/2201.12086) by Salesforce Research
- Powered by Pinecone vector database
- Built with FastAPI and Modal
