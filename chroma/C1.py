import chromadb
import pandas as pd
import uuid
from sentence_transformers import SentenceTransformer

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

parquet_file_path = "~/Desktop/FIB/CBDE/CBDE-vector-db/book-corpus-3.parquet"

load_data_to_chroma_in_batches(parquet_file_path)
