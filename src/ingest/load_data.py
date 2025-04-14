import pandas as pd

def load_reviews(path: str):
    return pd.read_excel(path, engine="openpyxl")
