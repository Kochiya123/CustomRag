import numpy as np
import psycopg2
from langchain_classic.chains import create_retrieval_chain

def read_column(cur):
    cur.execute('select id, name, description, price, image_url from Flower where vector IS NULL')
    if cur is None:
        print('No data found')
    return cur

def embedded_column(cur, conn, embed_model):
    try:
        rows = cur.fetchall()
        print(f"🧩 Found {len(rows)} rows to embed...")
        for row in rows:
            flower_id, name, description, price, image = row

            text = text = f"{name}. {description or ''}. Price: ${price:.2f}"

            text_embedding = embed_model.encode_text(
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