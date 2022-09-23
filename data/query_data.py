import sys
import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        if conn:
            print("succ")
    except Error as e:
        print(e)

    return conn

def select_all_messages(conn,tbl_nm):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    query = "SELECT * FROM "+tbl_nm
    cur.execute(query)

    rows = cur.fetchone()
    print(rows)
    # count = 0
    # for row in rows:
    #     print(row)
    #     count+=1
    #     if count > 10 :
    #         return


def main():
    if len(sys.argv) == 2:
        database_filepath = sys.argv[1]
        con = create_connection(database_filepath)
        tbl_name = "comb_table"
        select_all_messages(con,tbl_name)
    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python DisasterResponse.db')


if __name__ == '__main__':
    main()