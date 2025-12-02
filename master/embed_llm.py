import numpy as np
import psycopg2
from numpy.f2py.auxfuncs import throw_error

from master.connect import connect
from transformers import AutoModel
import torch
from huggingface_hub import notebook_login
import ast


threshold_score = 0.8

def read_column_vector_null(cur):
    cur.execute('select id, name, description, price, image_url from Flower where vector IS NULL')
    if cur is None:
        print('No data found')
    return cur

def read_all_column(cur):
    cur.execute('select id, name, description, price, image_url from Flower')
    if cur is None:
        print('No data found')
    return cur

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1))

local_dir = "../models/transformers/"

class Embed_llm:
    def __init__(self):
        self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4",cache_dir = local_dir, trust_remote_code=True, dtype = torch.float16)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        print("Model loaded")

    def retrieval_vector(self,cur,conn,query):
        try:
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

            top_3_index = np.array(
                np.argpartition(np.array(cosine_similarity(query_embedding, flower_vectors)).flatten(), -3)[-3:]) + 1
            top_3_index = top_3_index[::1]

            result = []
            for index in top_3_index:
                cur.execute("select name, description, price, image_url from Flower where id = %s", (int(index),))
                result.append(cur.fetchone())

        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            return 0
        return result

    #def embedded_all_column_vector_null(self, cur, conn):
        try:
            rows = read_column_vector_null(cur).fetchall()
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
            return 0
        return 1

    #def embedded_update_all_column(self, cur, conn):
        try:
            rows = read_all_column(cur).fetchall()
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
            return 0
        return 1

    #def embedded_single_column(self, cur, conn, id):
        try:
            cur.execute("select name, description, price, image_url from Flower where id = %s", (id,))
            row = cur.fetchone()

            name, description, price, image = row

            text = text = f"{name}. {description or ''}. Price: ${price:.2f}"

            text_embedding = self.model.encode_text(
                texts=text,
                task="retrieval",
                return_numpy=True,
            )
            text_embedding = text_embedding.squeeze().astype(np.float16).tolist()

            cur.execute('Update Flower Set vector = %s where id = %s', (text_embedding, id))

        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            return 0
        return id

    #def embedded_update_multiple_column(self, conn, cur, list_id):
        try:
            rows = []
            for id in list_id:
                cur.execute("select name, description, price, image_url from Flower where id = %s and vector is not null", (id,))
                row = cur.fetchone()
                rows.append(row)

                name, description, price, image = row

                text = text = f"{name}. {description or ''}. Price: ${price:.2f}"

                text_embedding = self.model.encode_text(
                    texts=text,
                    task="retrieval",
                    return_numpy=True,
                )
                text_embedding = text_embedding.squeeze().astype(np.float16).tolist()

                cur.execute('Update Flower Set vector = %s where id = %s', (text_embedding, id))

        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            return 0
        return list_id

    def embedded_add_single_column(self, cur, conn, product_string, image_url):
        try:
            text_embedding = []
            image_embedding = []

            if product_string:
                text_embedding = self.model.encode_text(
                    texts=product_string,
                    task="retrieval",
                    return_numpy=True,
                )
                text_embedding = text_embedding.squeeze().astype(np.float16).tolist()

            if image_url:
                image_embedding = self.model.encode_text(
                    texts=image_url,
                    task="retrieval",
                    return_numpy=True,
                )
                image_embedding = image_embedding.squeeze().astype(np.float16).tolist()

            cur.execute('Insert into product_vector (product_text, embedding_text, embedding_image) values (%s, %s, %s) returning id', (product_string, text_embedding, image_embedding))
            conn.commit()
        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            cur.execute("Rollback")
            conn.commit()
            return 0
        return cur.fetchone()

    def embedded_update_single_column(self, cur, conn, product_id, product_string, image_url):
        try:
            text_embedding = []
            image_embedding = []

            if product_string:
                text_embedding = self.model.encode_text(
                    texts=product_string,
                    task="retrieval",
                    return_numpy=True,
                )
                text_embedding = text_embedding.squeeze().astype(np.float16).tolist()

            if image_url:
                image_embedding = self.model.encode_text(
                    texts=image_url,
                    task="retrieval",
                    return_numpy=True,
                )
                image_embedding = image_embedding.squeeze().astype(np.float16).tolist()

            cur.execute('Update product_vector Set embedding_text = %s, embedding_image = %s where id = %s', (text_embedding, image_embedding, product_id))
            conn.commit()
        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            cur.execute("Rollback")
            conn.commit()
            return 0
        return product_id

    def embedded_delete_single_column(self, cur, conn, product_id):
        try:
            cur.execute("DELETE FROM product_vector WHERE id = %s RETURNING id", (product_id,))
            deleted = cur.fetchone()

            if deleted:
                return deleted[0]
            conn.commit()
        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            cur.execute("Rollback")
            conn.commit()
            return 0
        return product_id

    def retrieval_vector_new(self,cur,conn,query):
        try:
            query_embedding = np.array(self.model.encode_text(
                texts = query,
                task = "retrieval",
                return_numpy = True,
            )).reshape(1, -1)

            cur.execute("select embedding_text from product_vector where embedding_text IS NOT NULL")

            rows = cur.fetchall()

            flower_vectors = []
            for row in rows:
                vector = np.array(ast.literal_eval(row[0])).reshape(-1)
                flower_vectors.append(vector)

            flower_vectors = np.array(flower_vectors)

            top_3_index = np.array(
                np.argpartition(np.array(cosine_similarity(query_embedding, flower_vectors)).flatten(), -3)[-3:]) + 1
            top_3_index = top_3_index[::1]

            result = {}
            for index in top_3_index:
                cur.execute("select product_text from product_vector where id = %s", (int(index),))
                result[index] = cur.fetchone()[0]
            print("SUcess")
        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            return 0
        return result

