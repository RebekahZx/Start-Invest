"""Text encoders for founder bios, startup descriptions and investor theses.

Uses a sentence-transformer when one is installed and downloadable, otherwise
falls back to TF-IDF + truncated SVD (latent semantic analysis), which needs no
network access. Both return L2-normalised vectors, so a dot product is the
cosine similarity.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

log = logging.getLogger(__name__)
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextEncoder:
    def __init__(self, backend: str = "auto", dim: int = 96, seed: int = 7):
        self.requested = backend
        self.dim = dim
        self.seed = seed
        self.backend = "lsa"
        self._st = None
        self._tfidf: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    def fit(self, corpus: list[str]) -> "TextEncoder":
        if self.requested in ("auto", "sentence-transformers") and self._load_st():
            self.backend = "sentence-transformers"
            return self
        if self.requested == "sentence-transformers":
            raise RuntimeError(f"Could not load {ST_MODEL}")
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
        M = self._tfidf.fit_transform(corpus)
        k = min(self.dim, M.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(M)
        self.backend = "lsa"
        return self

    def _load_st(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._st = SentenceTransformer(ST_MODEL)
            return True
        except Exception as e:  # not installed, or no network
            log.info("sentence-transformers unavailable (%s); using LSA", e)
            return False

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "sentence-transformers":
            if self._st is None and not self._load_st():
                raise RuntimeError("sentence-transformers backend was used for training but is unavailable now")
            return np.asarray(self._st.encode(list(texts), normalize_embeddings=True))
        return normalize(self._svd.transform(self._tfidf.transform(list(texts))))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_st"] = None  # reloaded lazily by name
        return state
