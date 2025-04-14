from src.ingest.load_data import load_reviews
from src.embeddings.embedder import get_embedding_model, generate_embeddings
from src.vector_store.faiss_index import build_faiss_index
from src.summarizer.rag_pipeline import load_flan_model, rag_to_json
from src.utils.io_utils import save_json

# Step 1: Load data
df = load_reviews("Dataset_Kaggle_project.xlsx")

# Step 2: Generate embeddings
embed_model = get_embedding_model()
texts = df["Review"].tolist()
embeddings = generate_embeddings(embed_model, texts)

# Step 3: Build FAISS index
index = build_faiss_index(embeddings)

# Step 4: Load FLAN-T5
tokenizer, model = load_flan_model()

# Step 5: Run RAG summarization
query = "Disneyland Parks"
result = rag_to_json(query, 3, embed_model, index, df, tokenizer, model)

# Step 6: Save summary
save_json(result, "summary_output.json")
