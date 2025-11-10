# Alternative service implementation using Modal for deployment.
import modal
from typing import List
import os

image = modal.Image.from_dockerfile("docker/Dockerfile")
app = modal.App("image-search")

# Secrets for Pinecone API key
@app.function(
    image=image,
    gpu="L40",
    secrets=[modal.Secret.from_name("pinecone-secret")],
)
@modal.web_endpoint(method="POST")
def search(query: dict):
    """
    Primary endpoint for meme search.
    Expects JSON: {"text": "description of the image"}
    Returns: {"results": [list of meme IDs]}
    """
    import torch
    import numpy as np
    from pinecone.grpc import PineconeGRPC as Pinecone
    from models.SigLIP import get_compiled_siglip_text, SigLipTokenizer
    from models.BLIP import compile_blip, BlipProcessorInstance
    from PIL import Image
    import io
    import base64

    text_query = query.get("text", "")
    if not text_query:
        return {"error": "No text query provided"}, 400

    # 1. Encode text using SigLIP
    text_model = get_compiled_siglip_text()
    tokens = SigLipTokenizer([text_query])
    tokens = tokens.to("cuda")

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_embedding = text_model(tokens)
        text_embedding = torch.nn.functional.normalize(text_embedding, dim=-1)
        text_embedding_np = text_embedding.cpu().numpy()[0]

    # 2. Query Pinecone for top 32 matches
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ.get("PINECONE_INDEX_NAME", "im-search"))

    search_results = index.query(
        vector=text_embedding_np.tolist(),
        top_k=32,
        include_metadata=True
    )

    if not search_results.matches:
        return {"results": []}

    # 3. Prepare candidates for reranking
    candidate_ids = [match.id for match in search_results.matches]
    candidate_images = []

    # Fetch images from metadata (assuming base64 encoded or URLs)
    for match in search_results.matches:
        # This assumes images are stored as base64 in metadata
        # Adjust based on your actual Pinecone metadata structure
        if "image" in match.metadata:
            img_data = base64.b64decode(match.metadata["image"])
            img = Image.open(io.BytesIO(img_data))
            candidate_images.append(img)
        elif "image_url" in match.metadata:
            # If storing URLs, you'd need to fetch them
            # For now, skip this implementation
            pass

    if not candidate_images:
        # If no images available for reranking, return initial results
        return {"results": candidate_ids}

    # 4. Rerank using BLIP model
    blip_model = compile_blip()

    # Process images and text in batches
    batch_size = 16
    all_itm_scores = []
    all_cosine_scores = []

    for i in range(0, len(candidate_images), batch_size):
        batch_images = candidate_images[i:i+batch_size]
        batch_texts = [text_query] * len(batch_images)

        # Prepare inputs
        inputs = BlipProcessorInstance(
            images=batch_images,
            text=batch_texts,
            return_tensors="pt",
            padding=True
        )

        input_ids = inputs.input_ids.to("cuda")
        pixel_values = inputs.pixel_values.to("cuda").half()
        attention_mask = inputs.attention_mask.to("cuda")

        with torch.no_grad(), torch.cuda.amp.autocast():
            itm_scores, cosine_sim = blip_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask
            )

            # Get ITM probabilities (softmax on the 2-class output)
            itm_probs = torch.softmax(itm_scores, dim=1)[:, 1]  # Probability of match

            # Extract diagonal for cosine similarity (text-image pairs)
            cosine_values = torch.diagonal(cosine_sim)

            all_itm_scores.extend(itm_probs.cpu().numpy().tolist())
            all_cosine_scores.extend(cosine_values.cpu().numpy().tolist())

    # 5. Compute final scores: 0.7 * ITM + 0.3 * cosine
    final_scores = [
        0.7 * itm + 0.3 * cos
        for itm, cos in zip(all_itm_scores, all_cosine_scores)
    ]

    # 6. Sort by final score and return IDs
    ranked_results = sorted(
        zip(candidate_ids[:len(final_scores)], final_scores),
        key=lambda x: x[1],
        reverse=True
    )

    result_ids = [id for id, score in ranked_results]

    return {"results": result_ids}