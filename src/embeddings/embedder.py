from sentence_transformers import SentenceTransformer

def get_embedding_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def generate_embeddings(model, texts):
    return model.encode(texts, show_progress_bar=True)
