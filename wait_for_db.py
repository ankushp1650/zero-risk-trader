# wait_for_db.py
import time
import os
import pymysql
from pymysql.err import OperationalError

DB_HOST = os.environ.get('DB_HOST', 'db')
DB_PORT = int(os.environ.get('DB_PORT', 3306))
DB_USER = os.environ.get('DB_USER', 'berlin')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'berlin@123')
DB_NAME = os.environ.get('DB_NAME', 'user_platform_db')


def wait_for_database():
    print(f"Waiting for database {DB_HOST}:{DB_PORT}...")

    while True:
        try:
            conn = pymysql.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME
            )
            conn.close()
            print("Database is ready! 🚀")
            break
        except OperationalError:
            print("Database not ready yet. Retrying in 2 seconds...")
            time.sleep(2)




if __name__ == "__main__":
    wait_for_database()
