import numpy as np
import psycopg2
from numpy.f2py.auxfuncs import throw_error


from master.connect import connect
from transformers import AutoModel
import torch
from huggingface_hub import notebook_login
import ast

from master.rerank_llm import Rerank

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

#def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1))

def cosine_similarity(query_vec, matrix):
    """
    Compute cosine similarity between a single query vector and a matrix of vectors.
    query_vec: shape (1, d) or (d,)
    matrix: shape (n, d)
    """
    query_vec = np.array(query_vec).reshape(1, -1)
    matrix = np.array(matrix)

    # Normalize
    query_norm = np.linalg.norm(query_vec)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    sims = np.dot(matrix, query_vec.flatten()) / (matrix_norms * query_norm)
    return sims  # shape (n,)

local_dir = "../models/transformers/"

class Embed_llm:
    def __init__(self):
        self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4",cache_dir = local_dir, trust_remote_code=True, dtype = torch.float16, revision="main")
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

            similarity = cosine_similarity(query_embedding, flower_vectors)

            for i in range(similarity.shape[0]):
                for j in range(similarity.shape[1]):
                    print(similarity[i,j])

            top_3_index = np.array(
                np.argpartition(np.array(similarity).flatten(), -3)[-3:]) + 1
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

    def embedded_add_single_column_product(self, cur, conn, product_string, product_name, product_id, category_id, price):
        try:
            text_embedding = []

            if product_string:
                text_embedding = self.model.encode_text(
                    texts=product_string,
                    task="retrieval",
                    return_numpy=True,
                )
                text_embedding = text_embedding.squeeze().astype(np.float16).tolist()

            cur.execute('Insert into product_vector (product_id, category_id, price, product_text, product_name, embedding_text) values (%s, %s, %s, %s, %s, %s)', (product_id, category_id, price, product_string, product_name, text_embedding))
            conn.commit()
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            # Properly rollback the transaction
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return 0
        return product_id

    def embedded_update_single_column_product(self, cur, conn, product_id, product_string, product_name, category_id, price):
        try:
            text_embedding = []

            if product_string:
                text_embedding = self.model.encode_text(
                    texts=product_string,
                    task="retrieval",
                    return_numpy=True,
                )
                text_embedding = text_embedding.squeeze().astype(np.float16).tolist()

            fields = []
            values = []

            if category_id:
                fields.append("category_id = %s")
                values.append(category_id)
            if product_name:
                fields.append("product_name = %s")
                values.append(product_name)
            if price:
                fields.append("price = %s")
                values.append(price)
            if text_embedding:
                fields.append("embedding_text = %s")
                values.append(text_embedding)

            # Only run if there are fields to update
            if fields:
                sql = f"UPDATE product_vector SET {', '.join(fields)} WHERE product_id = %s"
                values.append(product_id)
                cur.execute(sql, tuple(values))

            conn.commit()
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return 0
        return product_id

    def embedded_delete_single_column_product(self, cur, conn, product_id):
        try:
            cur.execute("DELETE FROM product_vector WHERE product_id = %s", (product_id,))
            conn.commit()
            return product_id
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return 0

    def retrieval_vector_product(self, cur, conn, query):
        try:
            query_embedding = np.array(self.model.encode_text(
                texts = query,
                task = "retrieval",
                return_numpy = True,
            )).reshape(1, -1)

            cur.execute("select embedding_text from product_vector where embedding_text IS NOT NULL")
            rows = cur.fetchall()

            flower_vectors = [np.array(ast.literal_eval(row[0])).reshape(-1) for row in rows]
            flower_vectors = np.array(flower_vectors)

            similarities = cosine_similarity(query_embedding, flower_vectors)

            # Get top 10 indices sorted by similarity
            top_10_indices = np.argsort(similarities)[-10:][::-1] + 1

            result = []
            for index in top_10_indices:
                cur.execute("select product_id, product_text from product_vector where id = %s", (int(index),))
                result.append(cur.fetchone())

        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            return 0
        return result

    def retrieval_vector_image(self, cur,conn,image_url):
        try:
            image_embedding = np.array(self.model.encode_text(
                images = image_url,
                task = "retrieval",
                return_numpy = True,
            )).reshape(1, -1)

            cur.execute("select embedding_image from product_vector where embedding_text IS NOT NULL")

            rows = cur.fetchall()

            flower_vectors = []
            for row in rows:
                vector = np.array(ast.literal_eval(row[0])).reshape(-1)
                flower_vectors.append(vector)

            flower_vectors = np.array(flower_vectors)

            top_3_index = np.array(
                np.argpartition(np.array(cosine_similarity(image_embedding, flower_vectors)).flatten(), -3)[-3:]) + 1
            top_3_index = top_3_index[::1]

            result = {}
            for index in top_3_index:
                cur.execute("select product_text from product_vector where id = %s", (int(index),))
                result[index] = cur.fetchone()

        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            return 0
        return result

    def embedded_add_single_category(self, cur, conn, category_id, category_text):
        try:
            category_embedding = np.array(self.model.encode_text(
                texts = category_text,
                task = "retrieval",
                return_numpy = True,
            ))
            category_embedding = category_embedding.squeeze().astype(np.float16).tolist()

            cur.execute('Insert into category_vector (category_id, category_name, category_embedding) values (%s, %s, %s) returning id', (category_id, category_text, category_embedding))
            id = cur.fetchone()[0]
            conn.commit()
            return id
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            # Properly rollback the transaction
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return 0

    def embedded_update_single_category(self, cur, conn, category_id, category_text):
        try:
            category_embedding = np.array(self.model.encode_text(
                texts=category_text,
                task="retrieval",
                return_numpy=True,
            ))
            category_embedding = category_embedding.squeeze().astype(np.float16).tolist()

            fields = []
            values = []

            if category_text:
                fields.append("category_name = %s")
                values.append(category_text)
            if category_embedding:
                fields.append("category_embedding = %s")
                values.append(category_embedding)

            # Only run if there are fields to update
            if fields:
                sql = f"UPDATE category_vector SET {', '.join(fields)} WHERE category_id = %s"
                values.append(category_id)
                cur.execute(sql, tuple(values))

            conn.commit()
            return category_id
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            # Properly rollback the transaction
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return 0

    def embedded_delete_single_column_category(self, cur, conn, category_id):
        try:
            cur.execute("DELETE FROM category_vector WHERE category_id = %s", (category_id,))
            conn.commit()
            return category_id
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return 0

    def embedded_add_delivery_information(self, cur, conn, delivery_text):
        try:
            delivery_information_embedding = np.array(self.model.encode_text(
                texts = delivery_text,
                task = "retrieval",
                return_numpy = True,
            ))
            delivery_information_embedding = delivery_information_embedding.squeeze().astype(np.float16).tolist()
            cur.execute("Insert into delivery_information (delivery_text, delivery_embedding) values (%s, %s) Returning delivery_id", (delivery_text, delivery_information_embedding))
            id = cur.fetchone()[0]
            conn.commit()
            return id
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return 0

    def embedded_update_delivery_information(self, cur, conn, delivery_id, delivery_text):
        try:
            delivery_embedding = np.array(self.model.encode_text(
                texts=delivery_text,
                task="retrieval",
                return_numpy=True,
            ))
            delivery_embedding = delivery_embedding.squeeze().astype(np.float16).tolist()

            fields = []
            values = []

            if delivery_text:
                fields.append("delivery_text = %s")
                values.append(delivery_text)
            if delivery_embedding:
                fields.append("delivery_embedding = %s")
                values.append(delivery_embedding)

            # Only run if there are fields to update
            if fields:
                sql = f"UPDATE delivery_information SET {', '.join(fields)} WHERE delivery_id = %s"
                values.append(delivery_id)
                cur.execute(sql, tuple(values))

            conn.commit()
            return delivery_id
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            # Properly rollback the transaction
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return 0

    def embedded_delete_delivery_information(self, cur, conn, delivery_id):
        try:
            cur.execute("DELETE FROM delivery_information WHERE delivery_id = %s", (delivery_id,))
            conn.commit()
            return delivery_id
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return 0

    #def embedded_add_single_voucher_information(self, cur, conn, voucher_id, amount, ends_at, max_discount_amount, min_order_value, percent, starts_at):
        try:
            cur.execute("Insert into voucher_vector (voucher_id, amount, ends_at, max_discount_amount, min_order_value, percent, starts_at) values %s, %s, %s, %s, %s, %s, %s, %s", (voucher_id, amount, ends_at, max_discount_amount, min_order_value, percent, starts_at))
            conn.commit()
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return 0
        return voucher_id

    #def embedded_update_single_voucher_information(self, cur, conn, voucher_id, amount, ends_at, max_discount_amount, min_order_value, percent, starts_at):
        try:
            fields = []
            values = []

            if amount:
                fields.append("amount = %s")
                values.append(amount)
            if ends_at:
                fields.append("ends_at = %s")
                values.append(ends_at)
            if max_discount_amount:
                fields.append("max_discount_amount = %s")
                values.append(max_discount_amount)
            if min_order_value:
                fields.append("min_order_value = %s")
                values.append(min_order_value)
            if percent:
                fields.append("percent = %s")
                values.append(percent)
            if starts_at:
                fields.append("starts_at = %s")
                values.append(starts_at)

            # Only run if there are fields to update
            if fields:
                sql = f"UPDATE voucher_vector SET {', '.join(fields)} WHERE voucher_id = %s"
                values.append(voucher_id)
                cur.execute(sql, tuple(values))

            conn.commit()
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            # Properly rollback the transaction
        try:
            conn.rollback()
        except Exception as rollback_error:
            print(f"Rollback error: {rollback_error}")
        return voucher_id

    #def embedded_delete_single_voucher_information(self, cur, conn, voucher_id):
        try:
            cur.execute("DELETE FROM voucher_vector WHERE voucher_id = %s", (voucher_id,))
            conn.commit()
            return voucher_id
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return 0

    def embedded_retrieve_products_information(self, cur, conn, query, preference, flower, price):
        try:
            products = []
            product_vector = []
            result = []
            reranker = Rerank()
            if price & preference:
                cur.execute(f"select product_id, product_text from product_vector where price {preference} %s",price)
                products = cur.fetchall()
            elif price:
                cur.execute("select product_id, product_text from product_vector where price = %s and ",price)
                products = cur.fetchall()
            if flower:
                product_vector = self.retrieval_vector_product(cur, conn, query)

            if products & product_vector:
                result = reranker.combine_and_rerank_together(query, products, product_vector)
            elif products:
                result = reranker.rerank_query(query, products)
            elif product_vector:
                result = reranker.rerank_query(query, product_vector)
            else:
                raise Exception('Cannot retrieve result from product vector table!')
            return result
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            return 0

    def retrieve_random_products(self, cur, conn, limit=5):
        """
        Retrieve random products from the database for general questions.
        Returns a list of tuples: (product_name, product_text, price)
        """
        try:
            # Use ORDER BY RANDOM() to get random products
            # Limit to 5 products for general questions
            cur.execute(
                "SELECT product_name, product_text, price FROM product_vector "
                "WHERE product_name IS NOT NULL AND product_text IS NOT NULL "
                "ORDER BY RANDOM() LIMIT %s",
                (limit,)
            )
            products = cur.fetchall()
            return products if products else []
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Database error in retrieve_random_products: {error}")
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return []

    def embedded_retrieve_category_information(self, cur, conn, query, category):
        try:
            products = []
            results = []
            reranker = Rerank()
            if category:
                cur.execute("select product_id, product_text from product_vector full join category on product_vector.category_id = category_id where category = %s",category)
                products = cur.fetchall()
            products_vector = self.retrieval_vector_product(cur, conn, query)
            if products_vector & products:
                results = reranker.combine_and_rerank_together(query, products_vector, products)
            elif products_vector:
                results = reranker.rerank_query(query, products_vector)
            return results
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            return 0

    def embedded_retrieve_delivery_information(self, cur, conn, query):
        try:
            delivery_embedding = self.general_embedding(query).reshape(1, -1)

            # Select both id and embedding to get the ID for later retrieval
            cur.execute("select id, delivery_embedding from delivery_information where delivery_embedding IS NOT NULL")
            delivery_vectors = cur.fetchall()

            if not delivery_vectors:
                return None

            # Extract IDs and embeddings separately
            delivery_ids = [row[0] for row in delivery_vectors]
            delivery_embeddings = [np.array(ast.literal_eval(row[1])).reshape(-1) for row in delivery_vectors]
            delivery_embeddings = np.array(delivery_embeddings)

            similarities = cosine_similarity(delivery_embedding, delivery_embeddings)

            # Get index of highest similarity
            best_idx = np.argmax(similarities)

            # Get the actual database ID
            best_db_id = delivery_ids[best_idx]

            # Retrieve from database
            cur.execute("SELECT * FROM delivery_information WHERE id = %s", (best_db_id,))
            result = cur.fetchone()
            return result
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            return None


    def general_embedding(self, text):
        text_embedding = np.array(self.model.encode_text(
            texts=text,
            task="retrieval",
            return_numpy=True,
        ))
        return text_embedding

