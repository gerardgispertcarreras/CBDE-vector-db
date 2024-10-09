import psycopg2
import time
from utils.config import load_config


def connect(func):
    """Connect to the PostgreSQL database server"""
    config = load_config()
    try:
        # connecting to the PostgreSQL server
        with psycopg2.connect(**config) as conn:
            start_time = time.time()
            result = func(conn)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"The script has taken {elapsed_time:.2f} seconds")
            return result
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)
