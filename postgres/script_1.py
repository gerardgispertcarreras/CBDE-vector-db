import pandas as pd
import uuid


def script_1(conn):
    df = pd.read_parquet("~/Desktop/CBDE/CBDE-vector-db/book-corpus-3.parquet")

    # Seed for reproducibility.
    seed = 42
    sub_df = df.sample(n=20000, random_state=seed).reset_index()
    print("Script 1 executed successfully.")

    # Pujar-ho a la DB.
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
    # Execute the query
    cur.execute(create_table_query)

    insert_query = """
    INSERT INTO sentences VALUES (%s, %s);
    """
    for _, row in sub_df.iterrows():
        id = uuid.uuid4()
        print(id)
        cur.execute(insert_query, (str(id), row["text"]))

    # Commit the changes to the database
    conn.commit()

    # Close the cursor and connection
    cur.close()
    conn.close()

    print("Table created successfully!")
