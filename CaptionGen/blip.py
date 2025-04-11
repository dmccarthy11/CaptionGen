from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

def load_blip():
    """
    Loads a pretrained Blip model and processor

    Returns: model, processor
    """
    # processor = BlipProcessor.from_pretrained("model/blip-small-parquet")
    # model = BlipForConditionalGeneration.from_pretrained("model/blip-small-parquet")
    processor = BlipProcessor.from_pretrained("dmccarthy1145/BLIP-Instagram")
    model = BlipForConditionalGeneration.from_pretrained("dmccarthy1145/BLIP-Instagram")

    if torch.cuda.is_available():
        model.to("cuda")

    return model, processor

def greedy_caption(model, processor, image):
    """
    Captions an image using greedy search

    Args:
        - model: Blip pretrained model
        - processor: Blip processor
        - image: PIL Image. Expects to be opened already with PIL.Image.open(image) 

    Returns: caption
    """
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    out_greedy = model.generate(**inputs)
    return processor.decode(out_greedy[0], skip_special_tokens=True)

def generate_all_captions(model, processor, image):
    """
    Generates several captions for an image using different search techniques: greedy, beam, top_p, top_k

    Args:
        - model: Blip pretrained model
        - processor: Blip processor
        - image: PIL Image. Expects to be opened already with PIL.Image.open(image) 

    Returns: captions (list[strings]): a list of captions
    """
    # Preprocess image and generate multiple captions
    captions = []
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    # 1. Greedy decoding
    out_greedy = model.generate(**inputs)
    captions.append(processor.decode(out_greedy[0], skip_special_tokens=True))

    # 2. Beam search
    out_beam = model.generate(**inputs, num_beams=5, num_return_sequences=1)
    captions.append(processor.decode(out_beam[0], skip_special_tokens=True))

    # 3. Top-k sampling
    out_topk = model.generate(**inputs, do_sample=True, top_k=50, max_length=50)
    captions.append(processor.decode(out_topk[0], skip_special_tokens=True))

    # 4. Top-p (nucleus) sampling
    out_topp = model.generate(**inputs, do_sample=True, top_p=0.9, max_length=50)
    captions.append(processor.decode(out_topp[0], skip_special_tokens=True))

    return captions

