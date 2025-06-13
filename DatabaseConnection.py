import psycopg2

class DatabaseConnection:
    def __init__(self, user, password, dbname, host, port):
        self.conn = psycopg2.connect(
            user=user,
            password=password,
            dbname=dbname,
            host=host,
            port=port
        )
        self.conn.autocommit = True
        self.cur = self.conn.cursor()

    def close(self):
        self.cur.close()
        self.conn.close()
