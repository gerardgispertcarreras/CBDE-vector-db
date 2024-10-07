import chromadb
import pandas as pd
import uuid

def find_similar_sentences_with_chroma(parquet_file):
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="sentence_collection")

    df = pd.read_parquet(parquet_file).head(1000)
    seed = 42
    selected_sentences = df['text'].drop_duplicates().sample(n=10, random_state=seed).reset_index(drop=True)

    ids = []
    documents = []

    for _, row in df.iterrows():
        ids.append(str(uuid.uuid4()))
        documents.append(row["text"])

    collection.add(
        ids=ids,
        documents=documents
    )

    # Buscar las dos oraciones más similares para cada una de las 10 seleccionadas
    results = []
    for sentence in selected_sentences:
        cosine_result = collection.query(
            query_texts=[sentence],
            n_results=3
        )

        cosine_top_2 = [doc for doc in cosine_result['documents'][1:3]]
        results.append({
            "original_sentence": sentence,
            "cosine_top_2": cosine_top_2
        })

    # Mostrar los resultados
    for result in results:
        print("Original Sentence: ", result["original_sentence"])
        print("  - Top 2 Similar Sentences (Cosine Distance):")
        for idx, sim_sentence in enumerate(result["cosine_top_2"], start=1):
            print(f"    {idx}: {sim_sentence}")
        print("\n")

find_similar_sentences_with_chroma("~/Desktop/FIB/CBDE/CBDE-vector-db/book-corpus-3.parquet")
