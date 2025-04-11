from transformers import BlipProcessor, BlipForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
from PIL import Image
from datasets import load_dataset, Dataset
from evaluate import load
from itertools import islice
import nltk
from nltk.tokenize import word_tokenize
import os
os.environ["WANDB_DISABLED"] = "true"

TRAIN = False

processor = BlipProcessor.from_pretrained("./blip-small-parquet")
model = BlipForConditionalGeneration.from_pretrained("./blip-small-parquet")

if torch.cuda.is_available():
    model.to("cuda")

ds = load_dataset("Obscure-Entropy/ImageCaptioning_SmallParquets", 
                  split="train", 
                  streaming=True)

# Preprocessing function
def preprocess(example):
    # Convert image to RGB in case it's not in that format
    image = example["img"].convert("RGB")
    caption = example["en_cap"]  # Get the caption text

    # Use the processor to process both image and caption
    inputs = processor(images=image, text=caption, return_tensors="pt", padding="max_length", truncation=True, max_length=64)

    # Return pixel_values (processed image tensor) and input_ids (tokenized caption)
    return {
        "pixel_values": inputs["pixel_values"].squeeze(0),  # Image tensor
        "input_ids": inputs["input_ids"].squeeze(0),        # Tokenized caption
        "labels": inputs["input_ids"].squeeze(0)            # Tokenized caption
    }

# Apply preprocessing
small_ds = ds.take(10000)
processed_ds = [preprocess(sample) for sample in small_ds]

# Create a data collator for language modeling
data_collator = DataCollatorForSeq2Seq(processor.tokenizer, model=model, padding=True)

# Define data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.tokenizer,
    model=model,
    padding=True,
    return_tensors="pt"
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./blip-finetuned-captioning",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-6, 
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    evaluation_strategy="no",
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_ds,
    data_collator=data_collator
)

# If training mode set, start training
if TRAIN:
    trainer.train()

    # Save fine-tuned model
    model.save_pretrained("./blip-small-parquet")
    processor.save_pretrained("./blip-small-parquet")

# Else, evaluate
else:
    # Print a sample to check caption
    model.eval()

    # Load evaluation metrics
    nltk.download('punkt')
    rouge = load("rouge")
    bleu = load("bleu")

    # Take evaluation dataset and apply preprocessing
    eval_ds = list(islice(ds, 10000, 11000))
    eval_dataset = [preprocess(sample) for sample in eval_ds]
    references = [example["input_ids"] for example in eval_dataset]

    # Get predictions
    predictions = trainer.predict(eval_dataset)

    # Take token IDs and decode into text
    pred_ids = predictions.predictions
    decoded_preds = processor.batch_decode(pred_ids, skip_special_tokens=True)

    # Calculate ROUGE scores
    rouge_result = rouge.compute(predictions=decoded_preds, references=references)
    print("ROUGE:")
    for k, v in rouge_result.items():
        print(f"{k}: {v:.4f}")

    # Calculate BLEU scores
    tokenized_preds = [word_tokenize(pred) for pred in decoded_preds]
    tokenized_refs = [[word_tokenize(ref)] for ref in references]

    bleu_result = bleu.compute(predictions=tokenized_preds, references=tokenized_refs)
    print("\nBLEU:")
    print(f"BLEU: {bleu_result['bleu']:.4f}")