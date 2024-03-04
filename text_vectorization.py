import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


class Embedder:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def ave_pool(self, lhs, mask):
        last_hidden_state = lhs.masked_fill(~mask[..., None].bool(), 0.0)
        return last_hidden_state.sum(dim=1) / mask.sum(dim=1)[..., None]

    def get_embeddings(self, df, batch_size: int = 32):
        sentences = df["text"].astype(str).tolist()

        loader = DataLoader(sentences, batch_size=batch_size, shuffle=False)
        embeddings = []

        for batch in tqdm(loader, desc="Processing batches"):
            input = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**input).last_hidden_state

            attention_mask = input.attention_mask

            embedding = self.ave_pool(output, attention_mask).cpu().numpy()
            embeddings.extend(embedding)

        df["embedding"] = embeddings

        return df
