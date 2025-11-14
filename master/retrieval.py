import numpy as np
import psycopg2

from master.connect import connect


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1))

import ast
def retrieval_vector(query, model):
    try:
        cur, conn = connect()
        query_embedding = np.array(model.encode_text(
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