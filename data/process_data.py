import sys

import numpy as np
import pandas as pd
import sqlite3
def select_all_messages(conn,tbl_nm):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    query = "SELECT * FROM "+tbl_nm
    cur.execute(query)

    row = cur.fetchonr()
    print(row)

def create_table(database_filepath,df,tbl_name):
    conn = sqlite3.connect(database_filepath)

    print("Creating table in :" + database_filepath)
    df.to_sql(name=tbl_name, con=conn, if_exists='replace')

    conn.close()


def load_data(messages_filepath, categories_filepath):
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    #checking for nulls
    nulls = df_categories['categories'].isna().sum()
    print(nulls)

    col_split = df_categories['categories'].str.split('[;-]')
    rows = df_categories['categories'].str.split(';')
    arrOarr = []
    # count = 0
    for row in rows:
        row_arr = []
        for vals in row:
            num = vals[-1]
            # print(num)
            row_arr.append(num)
        arrOarr.append(row_arr)

    col_split_row = col_split[0]
    columns = []
    for col in col_split_row:
        if not (col.isdigit()):
            columns.append(col)
    dat = pd.DataFrame(arrOarr, columns=columns)
    df_categories = pd.concat([df_categories, dat], axis=1)
    print(df_categories.head())
    # print(map(lambda x: x[:-1], test))

    return df_messages, df_categories

def clean_data(df):
    merged_df = pd.concat([df[0], df[1]], axis=1)
    merged_df.drop(merged_df.columns[4],axis=1, inplace=True)
    # print(merged_df.head())
    return df,merged_df


def save_data(df, df_merged, database_filepath):
    #Categories table
    create_table(database_filepath,df[0],"Categories")
    #Messages table
    create_table(database_filepath,df[1],"Messages")
    #Concated table

    # print(merged_df.iloc[:,4])
    create_table(database_filepath,df_merged,"comb_table")

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(categories_filepath, messages_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df, df_merged = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, df_merged, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()