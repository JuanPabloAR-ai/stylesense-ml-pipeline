from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

def identity(x):
    """Pickle-safe identity function (use instead of lambdas in sklearn)."""
    return x

def join_text_cols(X) -> np.ndarray:
    """
    Join multiple text columns row-wise, robust to both pandas.DataFrame and numpy arrays.
    Ensures output shape == n_samples.
    """
    # If it's a DataFrame: join across columns per row
    if isinstance(X, pd.DataFrame):
        return (
            X.fillna("")
             .astype(str)
             .agg(" ".join, axis=1)
             .to_numpy()
        )

    # Else, coerce to numpy array (object) and handle 1D/2D
    X = np.asarray(X, dtype=object)

    # 1D: already one text column; just stringify safely
    if X.ndim == 1:
        return np.array(["" if pd.isna(x) else str(x) for x in X], dtype=object)

    # 2D: join across columns per row
    out = []
    for row in X:
        parts = []
        for w in row:
            parts.append("" if pd.isna(w) else str(w))
        out.append(" ".join(parts).strip())
    return np.array(out, dtype=object)

class SpacyLemmatizer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible lemmatizer using spaCy.
    - Input: iterable of strings
    - Output: list of tokens (lemmas) per document

    Notes for parallel/GridSearchCV:
    - Make pickling safe by not serializing the loaded nlp object.
    - Load the model lazily when needed.
    """
    def __init__(self, model: str = "en_core_web_sm", lowercase: bool = True):
        self.model = model
        self.lowercase = lowercase
        self._nlp = None

    # Make the estimator pickle-friendly by dropping _nlp
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_nlp"] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_loaded(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.model, disable=["parser", "ner", "textcat"])
            except OSError as e:
                raise OSError(
                    "spaCy model not found. Install with:\n"
                    "    python -m spacy download en_core_web_sm"
                ) from e

    def fit(self, X, y=None):
        self._ensure_loaded()
        return self

    def transform(self, X):
        self._ensure_loaded()
        texts = ["" if x is None else str(x) for x in X]
        docs = self._nlp.pipe(texts, batch_size=1000)
        output: List[List[str]] = []
        for doc in docs:
            toks: List[str] = []
            for t in doc:
                if t.is_space or t.is_punct or t.like_num or t.is_stop:
                    continue
                lemma = t.lemma_.strip()
                if self.lowercase:
                    lemma = lemma.lower()
                if lemma:
                    toks.append(lemma)
            output.append(toks)
        return output

# Ready-to-use transformers
text_joiner = FunctionTransformer(join_text_cols, validate=False)
