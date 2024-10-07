from sentence_transformers import SentenceTransformer
import numpy as np

BATCH_SIZE = 1000


def manhattan_distance(vector1, vector2):
    """Calculate the Manhattan distance between two vectors."""
    return np.sum(np.abs(np.array(vector1) - np.array(vector2)))


def euclidean_distance(vector1, vector2):
    """Calculate the Euclidean distance between two vectors."""
    return np.sqrt(np.sum(np.square(np.array(vector1) - np.array(vector2))))


def script_3(conn):
    print("Script 2 starting.")
    cur = conn.cursor()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sentence = "The dog is crazy, I'm a rainbow."
    sentence_embedding = model.encode(sentence)

    select_query = """
    SELECT * FROM sentences;
    """
    cur.execute(select_query)

    distances = []
    rows = cur.fetchmany(size=BATCH_SIZE)

    while rows:
        for row in rows:
            if row[2]:  # Assuming row[2] contains the vector
                dist = manhattan_distance(
                    row[2], sentence_embedding
                )  # Change to manhattan_distance if needed
                distances.append((dist, row[1]))  # (distance, sentence)
        rows = cur.fetchmany(size=BATCH_SIZE)

    # Sort distances and get the top 10
    top_similar_sentences = sorted(distances, key=lambda x: x[0])[:10]

    print("Top 10 most similar sentences:")
    for dist, target_sentence in top_similar_sentences:
        print(f"Distance: {dist:.4f}, Sentence: {target_sentence}")

    cur.close()
    print("Script 2 finished successfully.")
