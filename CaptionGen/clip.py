import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

def load_clip():
    """
    Loads a pretrained CLIP model and processor

    Returns: model, processor
    """
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if torch.cuda.is_available():
        clip_model.to("cuda")
    
    return clip_model, clip_processor

def select_best_caption_with_clip(clip_model, clip_processor, image, candidate_captions):
    """
    Selects the caption most similar to the image using CLIP.

    Args:
        clip_model: pretrained CLIP model
        clip_processor: CLIP processor to use
        image (PIL.Image): A PIL formatted image
        candidate_captions (List[str]): List of caption strings.

    Returns:
        (best_caption, all_scores): Tuple of the best caption and all similarity scores.
    """
    # Tokenize inputs
    inputs = clip_processor(text=candidate_captions, images=image, return_tensors="pt", padding=True).to(clip_model.device)

    # Get image/text embeddings
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds  # shape: (1, 512)
        text_embeds = outputs.text_embeds    # shape: (num_captions, 512)

    # Normalize
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # Cosine similarity
    similarity_scores = (image_embeds @ text_embeds.T).squeeze(0)  # shape: (num_captions,)

    # Select best
    best_idx = similarity_scores.argmax().item()
    best_caption = candidate_captions[best_idx]

    return best_caption, similarity_scores.tolist()