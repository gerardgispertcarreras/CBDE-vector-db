from collections import defaultdict
import time
import numpy as np
import pandas as pd
import uuid
from utils.chroma_connect import connect
from utils.constants import SENTENCES, NUM_SIMILAR_SENTENCES, BATCH_SIZE


def script_C2(client):
    print("Script C2 starting.")

    collection = client.get_collection(name="sentence_collection")
    time_by_sentence = []
    most_similar_sentences = defaultdict(list)
    for sentence in SENTENCES:
        start_time = time.time()
        most_similars = collection.query(
            query_texts=[sentence],
            n_results=NUM_SIMILAR_SENTENCES,
        )
        print(most_similars)
        end_time = time.time()
        time_by_sentence.append(end_time - start_time)
        for i in range(NUM_SIMILAR_SENTENCES):
            most_similar_sentences[sentence].append((
                most_similars["distances"][0][i],
                most_similars["documents"][0][i],
            ))

    print("\n[RESULTS]\n")
    for counter, (sentence, sentence_distance) in enumerate(
        most_similar_sentences.items()
    ):
        print(f'\n{counter + 1}. "{sentence}"')

        print("\nTop similarities:")
        for dist, target_sentence in sentence_distance:
            print(f'  - "{target_sentence}" with distance {dist:.4f}')

    min_time = np.min(time_by_sentence)
    max_time = np.max(time_by_sentence)
    avg_time = np.mean(time_by_sentence)
    std_dev_time = np.std(time_by_sentence)

    print(f"\nMinimum time: {min_time:.6f} seconds")
    print(f"Maximum time: {max_time:.6f} seconds")
    print(f"Average time: {avg_time:.6f} seconds")
    print(f"Standard Deviation: {std_dev_time:.6f} seconds")

    print("\nScript C2 finished successfully.\n")


if __name__ == "__main__":
    connect(script_C2)
