from __future__ import annotations
from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class Config:
    # Change these as needed for your file/columns
    path_csv: str = "data/Womens Clothing E-Commerce Reviews.csv"
    target: str = "Recommended IND"
    text_cols: List[str] = ("Review Text", "Title")
    num_cols: List[str] = ("Age", "Rating", "Positive Feedback Count")
    cat_cols: List[str] = ("Division Name", "Department Name", "Class Name")
    spacy_model: str = "en_core_web_sm"
    model_out: str = "models/model_pipeline.pkl"
    random_state: int = 42

def load_data(cfg: Config) -> pd.DataFrame:
    """Load CSV and ensure required columns exist; drop rows with missing target."""
    df = pd.read_csv(cfg.path_csv)
    needed = {cfg.target, *cfg.text_cols, *cfg.num_cols, *cfg.cat_cols}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    df = df.dropna(subset=[cfg.target])
    return df
