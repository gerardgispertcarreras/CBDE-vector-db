import psycopg2
from config import load_config

def connect(func):
    """ Connect to the PostgreSQL database server """
    config = load_config()
    try:
        # connecting to the PostgreSQL server
        with psycopg2.connect(**config) as conn:
            print('Connected to the PostgreSQL server.')
            return func(conn)
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)
