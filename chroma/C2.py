import statistics
import time
import chromadb
import pandas as pd
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine, euclidean

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="sentence_collection")

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(sentences, batch_size=500):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        batch_embeddings = model.encode(batch_sentences)
        embeddings.extend(batch_embeddings)
    return embeddings

def load_data_to_chroma_in_batches(parquet_file, batch_size=500):
    df = pd.read_parquet(parquet_file).head(10000).reset_index(drop=True)
    if 'text' not in df.columns:
        raise KeyError("La columna 'text' no se encuentra en el archivo Parquet.")
    
    sentences = df['text'].drop_duplicates().reset_index(drop=True)
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences.iloc[i:i + batch_size]
        
        ids = [str(uuid.uuid4()) for _ in range(len(batch_sentences))]
        embeddings = generate_embeddings(batch_sentences.tolist(), batch_size=batch_size)
        
        collection.add(
            ids=ids,
            documents=batch_sentences.tolist(),
            embeddings=[embedding.tolist() for embedding in embeddings]
        )
        print(f"Lote {i // batch_size + 1} cargado en ChromaDB")
query_times = []

def query_chroma_top_2(parquet_file, n_samples=10):
    df = pd.read_parquet(parquet_file).head(10000)
    sentences = df['text'].drop_duplicates().sample(n=n_samples, random_state=42).reset_index(drop=True)

    all_documents = collection.get(include=['documents', 'embeddings'])
    all_embeddings = np.array(all_documents['embeddings'])
    all_sentences = all_documents['documents']

    results = []
    for sentence in sentences:
        start_time = time.time()
        sentence_embedding = model.encode([sentence])[0]

        cosine_distances = [cosine(sentence_embedding, emb) for emb in all_embeddings]
        cosine_top_2_indices = np.argsort(cosine_distances)[1:3]  # Omitir la misma oración
        cosine_top_2 = [all_sentences[i] for i in cosine_top_2_indices]

        euclidean_distances = [euclidean(sentence_embedding, emb) for emb in all_embeddings]
        euclidean_top_2_indices = np.argsort(euclidean_distances)[1:3]
        euclidean_top_2 = [all_sentences[i] for i in euclidean_top_2_indices]

        end_time = time.time()
        query_times.append(end_time - start_time)

        results.append({
            "original_sentence": sentence,
            "cosine_top_2": cosine_top_2,
            "euclidean_top_2": euclidean_top_2
        })

    for result in results:
        print("Original Sentence:", result["original_sentence"])
        print("  - Top 2 Similar Sentences (Cosine Distance):")
        for idx, sim_sentence in enumerate(result["cosine_top_2"], start=1):
            print(f"    {idx}: {sim_sentence}")
        print("  - Top 2 Similar Sentences (Euclidean Distance):")
        for idx, sim_sentence in enumerate(result["euclidean_top_2"], start=1):
            print(f"    {idx}: {sim_sentence}")
        print("\n")

parquet_file_path = "~/Desktop/FIB/CBDE/CBDE-vector-db/book-corpus-3.parquet"

load_data_to_chroma_in_batches(parquet_file_path)
query_chroma_top_2(parquet_file_path)

print(f"\nEstadísticas de tiempo de collection.query:")
print(f"Tiempo máximo: {max(query_times):.4f} segundos")
print(f"Tiempo mínimo: {min(query_times):.4f} segundos")
print(f"Tiempo promedio: {sum(query_times) / len(query_times):.4f} segundos")
print(f"Desviación estándar: {statistics.stdev(query_times):.4f} segundos")