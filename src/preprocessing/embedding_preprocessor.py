from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingPreprocessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        # load pre-trained model(downloads once, caches locally)
        self.model = SentenceTransformer(model_name)

    def transform(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        "convert list of strings into (N, 384) dense embeddings"

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings

    @property
    def embedding_dim(self) -> int:
        return 384
