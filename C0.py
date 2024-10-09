import time
import numpy as np
import pandas as pd
import uuid
from utils.chroma_connect import connect
from utils.constants import NUM_SENTENCES, SEED, BATCH_SIZE


def script_C0(client):
    print("Script C0 starting.")

    distance_function = "cosine"  # "l2" or "cosine"
    try:
        client.delete_collection("sentence_collection")
    except Exception:
        pass
    collection = client.create_collection(name="sentence_collection",metadata={"hnsw:space": distance_function})
    df = pd.read_parquet("~/Desktop/CBDE/CBDE-vector-db/book-corpus-3.parquet")
    sub_df = df.sample(n=NUM_SENTENCES, random_state=SEED).reset_index()
    insertion_times = []
    for counter, (_, row) in enumerate(sub_df.iterrows()):
        id = uuid.uuid4()
        start_time = time.time()
        collection.add(ids=[str(id)], documents=[row["text"]])
        end_time = time.time()
        insertion_times.append(end_time - start_time)
        print(f"[{counter + 1}/{NUM_SENTENCES}]", end="\r")
    print("\033[K", end="\r")

    min_time = np.min(insertion_times)
    max_time = np.max(insertion_times)
    avg_time = np.mean(insertion_times)
    std_dev_time = np.std(insertion_times)

    print("\n[RESULTS]\n")
    print(f"Minimum time: {min_time:.6f} seconds")
    print(f"Maximum time: {max_time:.6f} seconds")
    print(f"Average time: {avg_time:.6f} seconds")
    print(f"Standard Deviation: {std_dev_time:.6f} seconds")

    print("\nScript C0 finished successfully.\n")

if __name__ == "__main__":
    connect(script_C0)
