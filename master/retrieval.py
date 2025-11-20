import numpy as np
import psycopg2
from master.connect import connect
from transformers import AutoModel
import torch
from huggingface_hub import notebook_login
import ast

class Embed_llm:
    def __init__(self):
        hf_token = "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"
        notebook_login()
        self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True, dtype = torch.float16)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1))


def retrieval_vector(query,self):
    try:
        cur, conn = connect()
        query_embedding = np.array(self.model.encode_text(
            texts = query,
            task = "retrieval",
            return_numpy = True,
        )).reshape(1, -1)

        cur.execute("select vector from Flower where vector IS NOT NULL")

        rows = cur.fetchall()

        flower_vectors = []
        for row in rows:
            vector = np.array(ast.literal_eval(row[0])).reshape(-1)
            flower_vectors.append(vector)

        flower_vectors = np.array(flower_vectors)

        top_3_index = np.array(np.argpartition(np.array(cosine_similarity(query_embedding, flower_vectors)).flatten(), -3)[-3:]) + 1
        top_3_index = top_3_index[::1]

        result = []
        for index in top_3_index:
            cur.execute("select name, description, price, image_url from Flower where id = %s", (int(index),))
            result.append(cur.fetchone())

        return result
    except(Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        conn.commit()
        conn.close()
        cur.close()
        print('Database connection closed.')

    return 0

def read_column(cur):
    cur.execute('select id, name, description, price, image_url from Flower where vector IS NULL')
    if cur is None:
        print('No data found')
    return cur

def embedded_all_column(cur, conn, self):
    try:
        rows = cur.fetchall()
        print(f"🧩 Found {len(rows)} rows to embed...")
        for row in rows:
            flower_id, name, description, price, image = row

            text = text = f"{name}. {description or ''}. Price: ${price:.2f}"

            text_embedding = self.model.encode_text(
                texts = text,
                task = "retrieval",
                return_numpy = True,
            )
            text_embedding = text_embedding.squeeze().astype(np.float16).tolist()

            cur.execute('Update Flower Set vector = %s where id = %s', (text_embedding, flower_id))
    except(Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        conn.commit()
        conn.close()
        cur.close()
        print('Database connection closed.')
    return 0