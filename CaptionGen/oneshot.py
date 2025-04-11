from blip import generate_all_captions
from clip import select_best_caption_with_clip

def oneshot_captions(blip_model, blip_processor, clip_model, clip_processor, image):
    """
    One-shot caption generation with CLIP to evaluate several captions from a BLIP model

    Returns: caption
    """
    captions = generate_all_captions(blip_model, blip_processor, image)

    best_caption, scores = select_best_caption_with_clip(clip_model, clip_processor, image, captions)

    return best_caption
