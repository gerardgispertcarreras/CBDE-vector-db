import time
import numpy as np
from sentence_transformers import SentenceTransformer
from constants import BATCH_SIZE, NUM_SENTENCES
from connect import connect

SELECT_QUERY = """
SELECT * FROM sentences;
"""

UPDATE_QUERY = """
UPDATE sentences
SET embedding = %s
WHERE id = %s
"""


def script_2(conn):
    print("Script 2 starting.")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    with conn.cursor() as cur_select:
        print("Retrieving rows...")
        cur_select.execute(SELECT_QUERY)
        print("Computing the embeddings and storing them in batches...")
        processed_rows = 0
        embedding_computation_times = []
        with conn.cursor() as cur_update:
            while rows := cur_select.fetchmany(size=BATCH_SIZE):
                for row in rows:
                    start_time = time.time()
                    embedding = model.encode(row[1])
                    cur_update.execute(UPDATE_QUERY, (embedding.tolist(), row[0]))
                    end_time = time.time()
                    embedding_computation_times.append(end_time - start_time)
                processed_rows += len(rows)
                print(f"[{processed_rows}/{NUM_SENTENCES}]", end="\r")
            print("\033[K", end="\r")

    conn.commit()

    min_time = np.min(embedding_computation_times)
    max_time = np.max(embedding_computation_times)
    avg_time = np.mean(embedding_computation_times)
    std_dev_time = np.std(embedding_computation_times)

    print("\n[RESULTS]\n")
    print(f"Minimum time: {min_time:.6f} seconds")
    print(f"Maximum time: {max_time:.6f} seconds")
    print(f"Average time: {avg_time:.6f} seconds")
    print(f"Standard Deviation: {std_dev_time:.6f} seconds")

    print("\nScript 2 finished successfully.\n")


if __name__ == "__main__":
    connect(script_2)
