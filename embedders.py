"""Airisto RAG - embedders.

Two backends behind one interface:

  HashingEmbedder            no install, deterministic, good for tests and typo-tolerant
                             matching, but does NOT understand meaning or Hinglish.
  SentenceTransformerEmbedder  real multilingual semantic model (pip install sentence-transformers).

Every embedder exposes:
    name: str                       stored in the index so mismatches are caught
    dim: int
    embed_documents(texts) -> list[list[float]]
    embed_query(text) -> list[float]
All vectors are L2-normalised, so cosine similarity == dot product.
"""
import math
import re
import zlib


def _l2_normalise(vec: list) -> list:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


class HashingEmbedder:
    """Character n-gram + word hashing into a fixed-size vector.

    Stable across runs (uses crc32, not Python's randomised hash()).
    """

    def __init__(self, dim: int = 512, ngram_range=(3, 5)):
        self.dim = dim
        self.ngram_range = ngram_range
        self.name = f"hashing-d{dim}-n{ngram_range[0]}{ngram_range[1]}"

    def _features(self, text: str):
        text = re.sub(r"\s+", " ", text.lower()).strip()
        for word in text.split(" "):
            if word:
                yield "w:" + word
        padded = f" {text} "
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            for i in range(len(padded) - n + 1):
                yield "c:" + padded[i : i + n]

    def embed_query(self, text: str) -> list:
        vec = [0.0] * self.dim
        for feat in self._features(text):
            vec[zlib.crc32(feat.encode("utf-8")) % self.dim] += 1.0
        vec = [math.log1p(x) for x in vec]  # dampen very frequent n-grams
        return _l2_normalise(vec)

    def embed_documents(self, texts: list) -> list:
        return [self.embed_query(t) for t in texts]


class SentenceTransformerEmbedder:
    """Multilingual semantic embeddings. Model is downloaded on first use.

    Some models (the e5 family) need prefixes: pass query_prefix='query: '
    and doc_prefix='passage: '. The MiniLM paraphrase models need none.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        query_prefix: str = "",
        doc_prefix: str = "",
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(model_name)
        # method was renamed in newer sentence-transformers; support both
        get_dim = getattr(self._model, "get_embedding_dimension", None) or \
            self._model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.name = f"st:{model_name}"

    def embed_documents(self, texts: list) -> list:
        vecs = self._model.encode(
            [self.doc_prefix + t for t in texts], normalize_embeddings=True
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list:
        vec = self._model.encode([self.query_prefix + text], normalize_embeddings=True)[0]
        return vec.tolist()


def get_embedder(kind: str = "hashing", model: str = "", query_prefix: str = "", doc_prefix: str = ""):
    if kind == "hashing":
        return HashingEmbedder()
    if kind in ("st", "sentence-transformers"):
        kwargs = {"query_prefix": query_prefix, "doc_prefix": doc_prefix}
        if model:
            kwargs["model_name"] = model
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"unknown embedder '{kind}' (use 'hashing' or 'st')")


def cosine(a: list, b: list) -> float:
    """Dot product; equals cosine similarity because vectors are normalised."""
    return sum(x * y for x, y in zip(a, b))
