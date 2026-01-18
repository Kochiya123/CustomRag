import mysql.connector
import os

def mysql_connect():
    try:
        print("Connecting to MySQL database...")
        return mysql.connector.connect(
            user=os.getenv('MYSQL_USERNAME'),
            password=os.getenv('MYSQL_PASSWORD'),
            host=os.getenv('MYSQL_HOST'),
            database=os.getenv('MYSQL_USERNAME')
        )
    except (Exception, mysql.connector.Error) as error:
        print(error)