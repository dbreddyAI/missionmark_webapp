import psycopg2
from globals import debug
import json


_connection = None
def get_connection(credentials_path="data/json/db_creds.json"):

    global _connection

    if not _connection:
        debug("Connecting to Postgres database...")

        creds = json.load(open(credentials_path, "r"))

        _connection = psycopg2.connect(host=creds["host"], dbname=creds["database"], user=creds["username"], password=creds["password"])
        debug(" -> Connection successful!", 1)

    return _connection
