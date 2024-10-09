from sentence_transformers import SentenceTransformer
import numpy as np
from connect import connect
from constants import BATCH_SIZE, SENTENCES, NUM_SIMILAR_SENTENCES
import time

SELECT_QUERY = """
SELECT * FROM sentences;
"""


def manhattan_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculate the Manhattan distance between two vectors."""
    return np.sum(np.abs(vector1 - vector2))


def euclidean_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculate the Euclidean distance between two vectors."""
    return np.linalg.norm(vector1 - vector2)


def cosine_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculate the Cosine distance between two vectors."""
    dot_product = np.dot(vector1, vector2)
    norm_a = np.linalg.norm(vector1)
    norm_b = np.linalg.norm(vector2)
    return 1 - (dot_product / (norm_a * norm_b)) if norm_a and norm_b else float("inf")


def script_3(conn) -> None:
    print("Script 3 starting.")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sentences_distances = {
        sentence: {
            "embedding": model.encode(sentence),
            "top_manhattan": [],
            "top_euclidean": [],
            "top_cosine": [],
        }
        for sentence in SENTENCES
    }

    similarity_times = []
    with conn.cursor() as cur:
        for sentence, sentence_distance in sentences_distances.items():
            cur.execute(SELECT_QUERY)
            while rows := cur.fetchmany(size=BATCH_SIZE):
                for row in rows:
                    start_time = time.time()
                    row_embedding = np.array(row[2])
                    sentence_embedding = sentence_distance["embedding"]
                    manhattan_dist = manhattan_distance(
                        row_embedding, sentence_embedding
                    )
                    euclidean_dist = euclidean_distance(
                        row_embedding, sentence_embedding
                    )
                    cosine_dist = cosine_distance(row_embedding, sentence_embedding)

                    sentence_distance["top_manhattan"].append((manhattan_dist, row[1]))
                    sentence_distance["top_euclidean"].append((euclidean_dist, row[1]))
                    sentence_distance["top_cosine"].append((cosine_dist, row[1]))
                    end_time = time.time()
                    similarity_times.append(end_time - start_time)

    print("\n[RESULTS]")
    for counter, (sentence, sentence_distance) in enumerate(
        sentences_distances.items()
    ):
        print(f'\n{counter + 1}. "{sentence}"')

        sentence_distance["top_manhattan"] = sorted(
            sentence_distance["top_manhattan"], key=lambda x: x[0]
        )[:NUM_SIMILAR_SENTENCES]
        sentence_distance["top_euclidean"] = sorted(
            sentence_distance["top_euclidean"], key=lambda x: x[0]
        )[:NUM_SIMILAR_SENTENCES]
        sentence_distance["top_cosine"] = sorted(
            sentence_distance["top_cosine"], key=lambda x: x[0]
        )[:NUM_SIMILAR_SENTENCES]

        print("\nTop similarities by Manhattan:")
        for dist, target_sentence in sentence_distance["top_manhattan"]:
            print(f'  - "{target_sentence}" with distance {dist:.4f}')

        print("\nTop similarities by Euclidean:")
        for dist, target_sentence in sentence_distance["top_euclidean"]:
            print(f'  - "{target_sentence}" with distance {dist:.4f}')

        print("\nTop similarities by Cosine:")
        for dist, target_sentence in sentence_distance["top_cosine"]:
            print(f'  - "{target_sentence}" with distance {dist:.4f}')

    min_time = np.min(similarity_times)
    max_time = np.max(similarity_times)
    avg_time = np.mean(similarity_times)
    std_dev_time = np.std(similarity_times)

    print(f"\nMinimum time: {min_time:.6f} seconds")
    print(f"Maximum time: {max_time:.6f} seconds")
    print(f"Average time: {avg_time:.6f} seconds")
    print(f"Standard Deviation: {std_dev_time:.6f} seconds")

    print("\nScript 3 finished successfully.\n")


if __name__ == "__main__":
    connect(script_3)
