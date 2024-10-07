from sentence_transformers import SentenceTransformer


BATCH_SIZE = 1000


def script_2(conn):
    print("Script 2 starting.")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cur = conn.cursor()
    select_query = """
    SELECT * FROM sentences;
    """
    print("Selecting rows.")
    cur.execute(select_query)
    rows = cur.fetchmany(size=BATCH_SIZE)
    update_query = """
        UPDATE sentences
        SET embedding = %s
        WHERE id = %s
    """
    while rows:
        ids = []
        embeddings = []
        sentences = []
        for row in rows:
            if not row[2]:
                ids.append(row[0])
                sentences.append(row[1])
        rows = cur.fetchmany(size=BATCH_SIZE)
        embeddings.extend(model.encode(sentences))
        for sentence_id, embedding in zip(ids, embeddings):
            cur.execute(update_query, (embedding.tolist(), sentence_id))


    conn.commit()
    cur.close()

    print("Script 2 finished successfully.")
