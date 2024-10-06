import chromadb
import pandas as pd
import uuid

def find_similar_sentences_with_chroma(parquet_file):
    # Inicializa el cliente de ChromaDB
    chroma_client = chromadb.Client()

    # Crea o carga una colección en ChromaDB para almacenar los vectores
    collection = chroma_client.get_or_create_collection(name="sentence_collection")

    # Cargar el archivo Parquet y seleccionar 10 oraciones diferentes
    df = pd.read_parquet(parquet_file)
    seed = 42  # Para reproducibilidad
    selected_sentences = df['text'].drop_duplicates().sample(n=10, random_state=seed).reset_index(drop=True)

    # Crear listas para IDs y documentos
    ids = []
    documents = []

    # Insertar todas las oraciones en ChromaDB con un ID único
    for _, row in df.iterrows():
        ids.append(str(uuid.uuid4()))  # Generar un ID único
        documents.append(row["text"])   # Agregar el texto

    # Insertar todos los documentos en la colección
    collection.add(
        ids=ids,
        documents=documents
    )

    # Buscar las dos oraciones más similares para cada una de las 10 seleccionadas
    results = []
    for sentence in selected_sentences:
        # Usar 'cosine' para el primer cálculo
        cosine_result = collection.query(
            query_embeddings=[sentence],
            n_results=3,  # El primero será la misma oración, y los otros dos serán los más similares
            distance_metric='cosine'
        )
        
        # Usar 'euclidean' para el segundo cálculo
        euclidean_result = collection.query(
            query_embeddings=[sentence],
            n_results=3,
            distance_metric='euclidean'
        )

        # Extraer las dos oraciones más similares (después de sí misma)
        cosine_top_2 = [doc['content'] for doc in cosine_result['documents'][1:3]]
        euclidean_top_2 = [doc['content'] for doc in euclidean_result['documents'][1:3]]

        # Guardar resultados
        results.append({
            "original_sentence": sentence,
            "cosine_top_2": cosine_top_2,
            "euclidean_top_2": euclidean_top_2
        })

    # Mostrar los resultados
    for result in results:
        print("Original Sentence: ", result["original_sentence"])
        print("  - Top 2 Similar Sentences (Cosine Distance):")
        for idx, sim_sentence in enumerate(result["cosine_top_2"], start=1):
            print(f"    {idx}: {sim_sentence}")
        print("  - Top 2 Similar Sentences (Euclidean Distance):")
        for idx, sim_sentence in enumerate(result["euclidean_top_2"], start=1):
            print(f"    {idx}: {sim_sentence}")
        print("\n")

# Uso de la función con un archivo Parquet
find_similar_sentences_with_chroma("~/Desktop/FIB/CBDE/CBDE-vector-db/book-corpus-3.parquet")
