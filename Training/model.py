from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os

# Hyperparameters
# Training
#   epochs
#   learning_rate
#   lr_scheduler_type: decaying lr, avoiding sudden drops
#   warmup_ratio: percent of training steps for a warmup to prevent large updates at the start
#   batch_size: dataset batch size
#   per_device_batch_size: training batch size per GPU (dependent on type of accelerator)
# Model Generation
#   max_length: max number of tokens to generate
#   temperature: higher temp increases randomness and creativity, but less reliable
#   top_p: filters unlikely words
#   repetition_penalty: reduces repeated phrases in outputs

# Set cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN = True

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
BlipModel = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
BlipModel.to(device)

# Function to get caption from an image
def generate_caption(image_path):
    # Load image (local path or URL)
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)

    # Preprocess the image and generate caption
    inputs = processor(images=image, return_tensors="pt")
    out = BlipModel.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Function to generate a post
def generate_post(caption, platform="Instagram"):
    # Customize prompt based on platform
    if platform == "Instagram":
        prompt = f"Create an engaging and creative Instagram caption for the following image description: {caption}"
    elif platform == "Twitter":
        prompt = f"Write a catchy tweet about this image description: {caption}"
    elif platform == "Blog":
        prompt = f"Write a blog post introduction about this image description: {caption}"
    elif platform == "generic":
        prompt = f"### Human: Write a social media photo caption based on this photo description: {caption}"

    # Tokenize input and generate text
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, do_sample=True, max_length=128, temperature=0.7, repetition_penalty=1.2, top_p=0.9)
    post = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return generated text (social media post)
    return post

if TRAIN:
    # Load pretrained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)

    tokenizer.pad_token = tokenizer.eos_token

else:
    # Load fine-tuned model
    model = GPT2LMHeadModel.from_pretrained("./Training/gpt2-caption-generator")
    tokenizer = GPT2Tokenizer.from_pretrained("./Training/gpt2-caption-generator")

# Load and prep dataset
ds = load_dataset("Waterfront/social-media-captions")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = ds.map(tokenize_function, batched=True, batch_size=32)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

os.environ["WANDB_DISABLED"] = "true"

# Create a data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model.resize_token_embeddings(len(tokenizer))  # Resize embeddings in case of new tokens

# Training configuration
training_args = TrainingArguments(
    output_dir="./gpt2-caption-generator",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save fine-tuned model
model.save_pretrained("./Training/gpt2-caption-generator-2")
tokenizer.save_pretrained("./Training/gpt2-caption-generator-2")

## Example usage
post = generate_post("a man walking two dogs in the park", platform="Instagram")
print("Generated Post:", post)

post = generate_post("a pizza party in the office", platform="Instagram")
print("Generated Post:", post)

post = generate_post("yoga outside in the sun", platform="Instagram")
print("Generated Post:", post)