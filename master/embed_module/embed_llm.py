import numpy as np
import psycopg2
import requests
import os

import ast

from master.rerank_module.rerank_llm import Rerank

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

class Embed_llm:
    def __init__(self):
        self.api_token = os.getenv("JINA_TOKEN")
        if not self.api_token:
            raise ValueError("JINA_TOKEN environment variable is not set")
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.model_name = "jina-embeddings-v4"
        print("Jina Embeddings API initialized")

    def encode_text(self, texts, task="retrieval.passage"):
        """
        Encode text(s) using Jina Embeddings v4 API.

        Args:
            texts: str or list of str - text(s) to encode
            task: str - task type (default: "retrieval.passage" for documents, use "retrieval.query" for queries)
            return_numpy: bool - whether to return numpy array (default: True)

        Returns:
            numpy array of embeddings
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": texts,
            "model": self.model_name,
            "task": task
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            embedding = np.array(result["data"][0]["embedding"])
            return embedding.tolist()
        except requests.exceptions.RequestException as e:
            print(f"Error calling Jina API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            raise

    def encode_image(self, image_url):
        """
        Encode an image from URL using Jina Embeddings v4 API.
        Passes the image URL directly to the API without processing.
        Returns the image embedding vector.
        """
        try:
            # Call Jina API with image URL directly
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            # Create image payload with URL
            image_payload = {
                "input": [image_url],
                "model": self.model_name,
                "task": "retrieval.passage"
            }
            
            api_response = requests.post(self.api_url, json=image_payload, headers=headers, timeout=30)
            api_response.raise_for_status()
            result = api_response.json()
            
            # Extract embedding from response
            embedding = np.array(result["data"][0]["embedding"])
            return embedding.tolist()
            
        except Exception as e:
            print(f"Error processing image from URL {image_url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None

    def retrieval_vector(self,cur,conn,query):
        try:
            query_embedding = np.array(self.encode_text(
                texts = query,
                task = "retrieval.query",
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

                text_embedding = self.encode_text(
                    texts = text,
                    task = "retrieval.passage",
                    return_numpy = True,
                )
                text_embedding = text_embedding

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

                text_embedding = self.encode_text(
                    texts = text,
                    task = "retrieval.passage",
                    return_numpy = True,
                )
                text_embedding = text_embedding

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

            text_embedding = self.encode_text(
                texts=text,
                task="retrieval.passage",
                
            )
            text_embedding = text_embedding

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

                text_embedding = self.encode_text(
                    texts=text,
                    task="retrieval.passage",
                    
                )
                text_embedding = text_embedding

                cur.execute('Update Flower Set vector = %s where id = %s', (text_embedding, id))

        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            return 0
        return list_id

    def embedded_add_single_column_product(self, cur, conn, product_string, product_name, product_id, category_id, price):
        try:

            text_embedding = self.encode_text(
                texts=product_string,
                task="retrieval.passage",
            )

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
                text_embedding = self.encode_text(
                    texts=product_string,
                    task="retrieval.passage",
                )

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
            query_embedding = self.encode_text(
                texts = query,
                task = "retrieval.query",
            )
            threshold = 0.6
            limit = 5
            query = """
                        SELECT * FROM (
                            SELECT 
                                id, 
                                product_id, 
                                product_text,
                                1 - (embedding_text <=> %s::vector) as similarity
                            FROM product_vector
                            WHERE embedding_text IS NOT NULL
                        ) subquery
                        WHERE similarity >= %s
                        ORDER BY similarity DESC
                        LIMIT %s
                    """
            cur.execute(query, (query_embedding, threshold, limit))
            rows = cur.fetchall()

            # Query database to get product_text for each product_id
            result = []
            for row in rows:
                id, product_id, text, similarity = row
                combined_result = [
                    product_id,
                    text,
                    float(similarity),
                ]
                result.append(combined_result)

            return result
        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            return []
        except Exception as e:
            print(f"Error in retrieval_vector_product: {e}")
            return []

    def embedded_add_single_category(self, cur, conn, category_id, category_text):
        try:
            category_embedding = self.encode_text(
                texts = category_text,
                task = "retrieval.passage",
            )

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
            category_embedding = self.encode_text(
                texts=category_text,
                task="retrieval.passage",
                
            )
            category_embedding = category_embedding

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
            delivery_information_embedding = np.array(self.encode_text(
                texts = delivery_text,
                task = "retrieval.passage",
            ))
            delivery_information_embedding = delivery_information_embedding
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
            delivery_embedding = np.array(self.encode_text(
                texts=delivery_text,
                task="retrieval.passage",
                
            ))
            delivery_embedding = delivery_embedding

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
            convert_price = str(price)
            products = []
            product_vector = []
            result = []
            reranker = Rerank()
            if price is not None and preference is not None:
                if preference.lower() == "between":
                    price_min = int(price * 0.8)
                    price_max = int(price * 1.2)
                    sql = "SELECT product_id, product_text FROM product_vector WHERE price BETWEEN %s AND %s order by price desc limit 5"
                    cur.execute(sql, (price_min, price_max))
                    products = cur.fetchall()
                else:
                    cur.execute(f"select product_id, product_text from product_vector where price {preference} %s order by price limit 5", (
                        convert_price,))
                    products = cur.fetchall()
            elif price:
                cur.execute("select product_id, product_text from product_vector where price = %s limit 5", (convert_price,))
                products = cur.fetchall()
            else:
                product_vector = self.retrieval_vector_product(cur, conn, query)

            if product_vector and products:
                result = reranker.combine_and_rerank_together(query, products, product_vector)
            elif product_vector:
                return product_vector
            elif products:
                return reranker.rerank_query(query, products)
            else:
                raise Exception('Cannot retrieve result from product vector table!')
            return result
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            return 0

    def retrieve_random_products(self, cur, conn, limit=5):
        """
        Retrieve random products from the database without embedding.
        Returns a list of tuples: (product_id, product_name, product_text, price)
        """
        try:
            # Use ORDER BY RANDOM() to get random products
            cur.execute(
                "SELECT product_id, product_name, product_text, price FROM product_vector "
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

    def list_all_categories(self, cur, conn):
        """
        Retrieve all categories from the database without embedding.
        Returns a list of tuples: (category_id, category_name)
        """
        try:
            cur.execute(
                "SELECT category_id, category_name FROM category_vector "
                "WHERE category_name IS NOT NULL "
                "ORDER BY category_name"
            )
            categories = cur.fetchall()
            return categories if categories else []
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Database error in list_all_categories: {error}")
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return []

    def retrieve_products_by_category(self, cur, conn, category_name):
        """
        Retrieve all products for a specific category without embedding.
        Returns a list of tuples: (product_id, product_name, product_text, price)
        """
        try:
            # First, get the category_id from category_name
            cur.execute(
                "SELECT category_id FROM category_vector WHERE LOWER(category_name) = LOWER(%s)",
                (category_name,)
            )
            category_result = cur.fetchone()
            
            if not category_result:
                return []
            
            category_id = category_result[0]
            
            # Get all products for this category
            cur.execute(
                "SELECT product_id, product_name, product_text, price FROM product_vector "
                "WHERE category_id = %s AND product_name IS NOT NULL AND product_text IS NOT NULL",
                (category_id,)
            )
            products = cur.fetchall()
            return products if products else []
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Database error in retrieve_products_by_category: {error}")
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return []

    def embedded_retrieve_category_information(self, cur, conn, query, category):
        """
        Retrieve products by category with embedding-based reranking.
        This method combines category filtering with vector similarity search.
        """
        try:
            products = []
            results = []
            reranker = Rerank()
            
            if category:
                # Get category_id from category_name
                cur.execute(
                    "SELECT category_id FROM category_vector WHERE LOWER(category_name) = LOWER(%s)",
                    (category,)
                )
                category_result = cur.fetchone()
                
                if category_result:
                    category_id = category_result[0]
                    # Get all products for this category
                    cur.execute(
                        "SELECT product_id, product_text FROM product_vector WHERE category_id = %s",
                        (category_id,)
                    )
                    products = cur.fetchall()
            # Also do vector-based retrieval for the query
            products_vector = self.retrieval_vector_product(cur, conn, query)
            
            # Combine and rerank if both exist
            if products_vector and products:
                results = reranker.combine_and_rerank_together(query, products_vector, products)
            elif products_vector:
                return products_vector
            elif products:
                # If only category products exist, rerank them
                results = reranker.rerank_query(query, products)
            else:
                return []
            
            return results
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error in embedded_retrieve_category_information: {error}")
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return []

    def embedded_add_general_information(self, cur, conn, general_text):
        try:
            general_embedding = self.encode_text(
                texts = general_text,
                task = "retrieval.passage",
            )
            cur.execute("Insert into general_information (general_text, general_embedding) values (%s, %s) Returning general_id", (general_text, general_embedding))
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

    def embedded_update_general_information(self, cur, conn, general_id, general_text):
        try:
            general_embedding = np.array(self.encode_text(
                texts=general_text,
                task="retrieval.passage",
                
            ))
            general_embedding = general_embedding

            fields = []
            values = []

            if general_text:
                fields.append("general_text = %s")
                values.append(general_text)
            if general_embedding:
                fields.append("general_embedding = %s")
                values.append(general_embedding)

            # Only run if there are fields to update
            if fields:
                sql = f"UPDATE general_information SET {', '.join(fields)} WHERE general_id = %s"
                values.append(general_id)
                cur.execute(sql, tuple(values))

            conn.commit()
            return general_id
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            # Properly rollback the transaction
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return 0

    def embedded_delete_general_information(self, cur, conn, general_id):
        try:
            cur.execute("DELETE FROM general_information WHERE general_id = %s", (general_id,))
            conn.commit()
            return general_id
        except(Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            try:
                conn.rollback()
            except Exception as rollback_error:
                print(f"Rollback error: {rollback_error}")
            return 0

    def embedded_retrieve_general_information(self, cur, conn, query):
        try:
            query_embedding = self.encode_text(query)

            # Select both id and embedding to get the ID for later retrieval
            # Note: PostgreSQL vector type may need special handling
            cur.execute(
                "select general_id, general_text ,1- (general_embedding <=> %s::vector) as similarity from general_information where general_embedding IS NOT NULL order by similarity limit 5",
                (query_embedding,))
            rows = cur.fetchall()

            result = []
            for row in rows:
                general_id, text, similarity = row
                combined_result = [
                    general_id,
                    text,
                    float(similarity),
                ]
                result.append(combined_result)

            return result
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Database error: {error}")
            return None


    def general_embedding(self, text):
        text_embedding = np.array(self.encode_text(
            texts=text,
            task="retrieval.query",
            
        ))
        return text_embedding


    def compute_image_text_similarity(self, cur, conn, image_url):
        """
        Compute cosine similarity between image embedding and product text embeddings.
        Returns list of tuples: (product_id, product_text, similarity_score)
        """
        try:
            # Encode the image
            image_embedding = self.encode_image(image_url)
            
            # Get all product embeddings from database
            threshold = 0.6
            limit = 5
            query = """
                                    SELECT * FROM (
                                        SELECT 
                                            id, 
                                            product_id, 
                                            product_text,
                                            1 - (embedding_text <=> %s::vector) as similarity
                                        FROM product_vector
                                        WHERE embedding_text IS NOT NULL
                                    ) subquery
                                    WHERE similarity >= %s
                                    ORDER BY similarity DESC
                                    LIMIT %s
                                """
            cur.execute(query, (image_embedding, threshold, limit))
            rows = cur.fetchall()

            result = []
            for row in rows:
                id, product_id, text, similarity = row
                combined_result = [
                    product_id,
                    text,
                    float(similarity),
                ]
                result.append(combined_result)
            
            return result
        except Exception as e:
            print(f"Error computing image-text similarity: {e}")
            return []

