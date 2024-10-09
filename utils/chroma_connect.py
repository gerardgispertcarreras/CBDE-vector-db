import time
import chromadb


def connect(func):
    start_time = time.time()
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    func(chroma_client)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script has taken {elapsed_time:.2f} seconds")
