# Image Search Backend

This repo contains the code for the search backend of my custom image search engine.\
Images are fetched from Reddit on a schedule, bulk processed, and stored in the Pinecone vector database. That codebase can be found [here](https://github.com/mac-op/image-search-ingest).

## Overview

Users can describe an image in plain text (e.g., "a confused cat looking at a computer") and receive the most relevant images from the database. The search pipeline combines fast vector similarity search with sophisticated reranking for optimal results.

## Architecture

### Pipeline
Models are provided by Hugging Face, and compiled with TensorRT.
1. **Text Encoding**: User query is encoded using SigLIP text model
2. **Initial Retrieval**: Pinecone vector database returns top candidate matches
3. **Reranking**: BLIP model scores candidates using:
   - Image-Text Matching (ITM) score
   - Cosine similarity between embeddings
4. **Final Ranking**: Results ranked by `0.7 × ITM + 0.3 × cosine similarity`

### Prerequisites

- Python 3.12+
- CUDA-capable GPU
- Modal account
- Pinecone account

### Installation

1. Install dependencies:
```bash
pip install -r pyproject.toml
```

2. Set up Modal:
```bash
modal setup
```

3. Configure Pinecone secret in Modal:
```bash
modal secret create pinecone-secret \
  PINECONE_API_KEY=your_api_key \
  PINECONE_INDEX_NAME=image-search
```

### Deployment

Deploy to Modal:
```bash
modal deploy app.py
```

## Usage

### API Endpoint

**POST** `/search`

Request body:
```json
{
  "text": "a surprised pikachu face"
}
```

Response:
```json
{
  "results": ["image_id_1", "image_id_2", "image_id_3", ...]
}
```

## Acknowledgements
- [SigLIP](https://arxiv.org/abs/2303.15343) by Google Research
- [BLIP](https://arxiv.org/abs/2201.12086) by Salesforce Research
