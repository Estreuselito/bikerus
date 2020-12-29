import pandas as pd
import sqlite3
from sqlite3 import Error
import os


def create_connection(db_name):
    """create a database connection to the SQLite database specified by db_name

    Parameters
    ----------
    db_name: str
        database name

    Returns
    -------
    c: connection object
        conn.cursor

    conn: connection
        direct connect to database
    """
    conn = sqlite3.connect(db_name, check_same_thread=False)
    # c = conn.cursor()
    return conn


def table_exists(table_name, conn):
    """check if a table already exists within the database

    Parameters
    ----------
    table_name: str
        name of the table within the database

    conn: connection
        connection to database

    Returns
    -------
    True: when table already exists
    False: when table does not exists
    """
    c = conn.cursor()
    c.execute(
        '''SELECT count(name) FROM sqlite_master WHERE TYPE = 'table' AND name = '{}' '''.format(table_name))
    if c.fetchone()[0] == 1:
        return True
    return False


def check_and_create_and_insert(conn, table_name, df, sql_table_creating_string):
    """Check if table already exists, and when not create it

    Parameters
    ---------
    conn: connection
        Connection to database

    table_name: str
        the name of the table which shall be checked and/or created

    df: dataframe
        the dataframe, which should be saved into the database

    sql_table_createing_string: str
        a string literal of SQL command, in order to create a table    

    Returns
    -------
    None 
    """
    if table_exists(table_name, conn):
        print("Table already exists!")
        return None
    else:
        conn.execute(sql_table_creating_string)
        df.to_sql(name=table_name, con=conn, if_exists='append', index=False)
        print("Table is created!")
        return None


if not os.path.exists("./database"):
    os.makedirs("./database")

connection = create_connection("./database/BikeRental.db")
