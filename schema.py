
import sqlite3
conn = sqlite3.connect('test.db')

def get_tables():
    sql_query = """SELECT name FROM sqlite_master 
    WHERE type='table';"""
 
    # Creating cursor object using connection object
    cursor = conn.cursor()
     
    # executing our sql query
    cursor.execute(sql_query)
    return cursor.fetchall()

def sqlite_table_schema(conn, name):
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE name=?;", [name])
    sql = cursor.fetchone()[0]
    cursor.close()
    return sql

tables = list(map(lambda x: x[0], get_tables()))

for table in tables:
    print(sqlite_table_schema(conn, table))
