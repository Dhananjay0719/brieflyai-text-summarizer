import os
import torch
from transformers import pipeline

# performace and memory optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_grad_enabled(False)

# lightweight production-grade model
summarizer = pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6",
    tokenizer="sshleifer/distilbart-cnn-12-6",
    device=-1  # CPU safe (Render compatible)
)

def abstractive_summary(text):
    words = len(text.split())

    max_len = min(130, max(60, words // 2))
    min_len = max(30, max_len // 2)

    result = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True
    )

    return result[0]["summary_text"]
