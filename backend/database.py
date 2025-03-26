import psycopg2

def get_db_connection():
    return psycopg2.connect(
        dbname="real_estate",
        user="postgres",
        password="root",
        host="localhost",
        port="5432"
    )
