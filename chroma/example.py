import chromadb
import pandas as pd
import uuid

def find_similar_sentences_with_chroma(parquet_file):
    # Inicializa el cliente de ChromaDB
    chroma_client = chromadb.Client()

    # Crea o carga una colección en ChromaDB para almacenar los vectores
    # Define la métrica de distancia aquí si es posible
    collection = chroma_client.get_or_create_collection(name="sentence_collection")

    # Cargar solo las primeras 10,000 filas del archivo Parquet
    df = pd.read_parquet(parquet_file).head(1000)
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
        # Obtener las oraciones más similares usando la métrica por defecto (probablemente 'cosine')
        cosine_result = collection.query(
            query_texts=[sentence],
            n_results=3  # El primero será la misma oración, y los otros dos serán los más similares
        )

        # Extraer las dos oraciones más similares (después de sí misma)
        cosine_top_2 = [doc for doc in cosine_result['documents'][1:3]]

        # Guardar resultados
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

# Uso de la función con un archivo Parquet
find_similar_sentences_with_chroma("~/Desktop/FIB/CBDE/CBDE-vector-db/book-corpus-3.parquet")
