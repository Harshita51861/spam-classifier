import re
import pandas as pd


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " emailaddr ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _to_label(x) -> int:
    return 1 if str(x).strip().lower() in ("spam", "1") else 0


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    if {"v1", "v2"}.issubset(df.columns):  # SMS Spam Collection format
        df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df[["label", "text"]].dropna().drop_duplicates()
    df["label"] = df["label"].map(_to_label)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]
    return df.reset_index(drop=True)