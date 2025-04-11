from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import remove_erraneous_chars

def load_gpt2():
    """
    Loads the fine-tuned gpt-2 model.

    Returns: model, tokenizer
    """
    # tokenizer = GPT2Tokenizer.from_pretrained("./model/gpt2")
    # model = GPT2LMHeadModel.from_pretrained("./model/gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("dmccarthy1145/GPT2-Instagram")
    model = GPT2LMHeadModel.from_pretrained("dmccarthy1145/GPT2-Instagram")

    return model, tokenizer

def generate_gpt2_caption(model, tokenizer, short_caption):
    prompt = f"Can you rewrite this caption to make it more fun and engaging for Instagram? {short_caption}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=60, do_sample=True, top_k=50, top_p=0.95)

    decoded_text = (tokenizer.decode(outputs[0], skip_special_tokens=True))

    # Strip prompt from output
    generated_caption = decoded_text[len(prompt):].strip()

    return remove_erraneous_chars(generated_caption)