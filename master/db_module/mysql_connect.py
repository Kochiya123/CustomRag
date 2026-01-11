import mysql.connector
import os

import psycopg2


def mysql_connect():
    try:
        print("Connecting to MySQL database...")
        return mysql.connector.connect(
            user=os.getenv('MYSQL_USERNAME'),
            password=os.getenv('MYSQL_PASSWORD'),
            host=os.getenv('MYSQL_HOST'),
            database=os.getenv('MYSQL_DATABASE')
        )
    except (Exception, mysql.connector.Error) as error:
        print(error)