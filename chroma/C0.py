import chromadb
import pandas as pd
import uuid
import time
import numpy as np

def c0(conn):
    chroma_client = chromadb.Client()

    collection = chroma_client.get_or_create_collection(name="my_collection")

    df = pd.read_parquet("~/Desktop/FIB/CBDE/CBDE-vector-db/book-corpus-3.parquet")

    seed = 42
    sub_df = df.sample(n=20000, random_state=seed).reset_index()
    print("Script 1 executed successfully.")

    cur = conn.cursor()

    drop_table_query = '''
    DROP TABLE IF EXISTS sentences;
    '''
    cur.execute(drop_table_query)

    create_table_query = """
    CREATE TABLE sentences (
        id UUID PRIMARY KEY,
        sentence TEXT
    );
    """
    cur.execute(create_table_query)

    insert_query = """
    INSERT INTO sentences (id, sentence) VALUES (%s, %s);
    """

    times = []

    for _, row in sub_df.iterrows():
        id = uuid.uuid4()
        sentence = row["text"]
        start_time = time.time()
        collection.add(documents=[{"id": str(id), "content": sentence}])
        cur.execute(insert_query, (str(id), sentence))
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)

    conn.commit()

    cur.close()
    conn.close()

    # Cálculo de estadísticas sobre los tiempos de inserción
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    avg_time = np.mean(times)

    print(f"Minimum time: {min_time:.4f} seconds")
    print(f"Maximum time: {max_time:.4f} seconds")
    print(f"Standard deviation: {std_time:.4f} seconds")
    print(f"Average time: {avg_time:.4f} seconds")
    print("Table created and data uploaded successfully!")

c0("~/Desktop/FIB/CBDE/CBDE-vector-db/book-corpus-3.parquet")