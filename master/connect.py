import os
import psycopg2
from master.config import config


def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))

        # create a cursor
        cur = conn.cursor()

        return cur, conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
