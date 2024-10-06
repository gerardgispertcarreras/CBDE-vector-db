import chromadb
import pandas as pd
import uuid

def script_1(conn):
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

    # Inserta los datos y añade las sentencias a la colección en ChromaDB
    for _, row in sub_df.iterrows():
        id = uuid.uuid4()
        sentence = row["text"]
        collection.add(documents=[{"id": str(id), "content": sentence}])
        cur.execute(insert_query, (str(id), sentence))

    conn.commit()

    cur.close()
    conn.close()

    print("Table created and data uploaded successfully!")

