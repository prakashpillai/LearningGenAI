from transformers import T5Tokenizer, T5ForConditionalGeneration
import json

def load_flan_model(model_name="google/flan-t5-small"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def rag_to_json(query, k, embed_model, index, df, tokenizer, model):
    query_vector = embed_model.encode([query], convert_to_numpy=True).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    top_k_reviews = [df.iloc[idx]['Review'] for idx in indices[0]]
    
    context = " ".join([review[:300] for review in top_k_reviews])
    prompt = f"Summarize the following hotel reviews : {context}"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=100)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = {
        "query": query,
        "summary": summary,
        "top_k_reviews": top_k_reviews
    }
    return result
