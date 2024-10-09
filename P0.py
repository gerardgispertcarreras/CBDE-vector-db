import pandas as pd
import uuid
import numpy as np
import time
from utils.constants import NUM_SENTENCES, SEED
from utils.connect import connect

CREATE_TABLE_QUERY = """
CREATE TABLE sentences (
    id UUID PRIMARY KEY,
    sentence TEXT,
    embedding FLOAT[]
);
"""

DROP_TABLE_QUERY = """
DROP TABLE IF EXISTS sentences;
"""

INSERT_QUERY = """
INSERT INTO sentences VALUES (%s, %s, NULL);
"""


def script_P0(conn):
    print("Script P0 starting.")

    df = pd.read_parquet("~/Desktop/CBDE/CBDE-vector-db/book-corpus-3.parquet")
    sub_df = df.sample(n=NUM_SENTENCES, random_state=SEED).reset_index()

    with conn.cursor() as cur:
        print("Dropping the table if exists...")
        cur.execute(DROP_TABLE_QUERY)

        print("Creating the table...")
        cur.execute(CREATE_TABLE_QUERY)

        insertion_times = []

        print("Inserting the sentences...")
        for _, row in sub_df.iterrows():
            id = uuid.uuid4()
            start_time = time.time()
            cur.execute(INSERT_QUERY, (str(id), row["text"]))
            end_time = time.time()
            insertion_times.append(end_time - start_time)

    conn.commit()

    min_time = np.min(insertion_times)
    max_time = np.max(insertion_times)
    avg_time = np.mean(insertion_times)
    std_dev_time = np.std(insertion_times)

    print("\n[RESULTS]\n")
    print(f"Minimum time: {min_time:.6f} seconds")
    print(f"Maximum time: {max_time:.6f} seconds")
    print(f"Average time: {avg_time:.6f} seconds")
    print(f"Standard Deviation: {std_dev_time:.6f} seconds")

    print("\nScript P0 finished successfully.\n")


if __name__ == "__main__":
    connect(script_P0)
