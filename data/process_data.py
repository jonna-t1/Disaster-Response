import sys
import sqlite3
import pandas as pd

"""
Uploading to upload csv data to database
"""

def select_all_messages(conn, tbl_nm):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    query = "SELECT * FROM " + tbl_nm
    cur.execute(query)

    row = cur.fetchonr()
    print(row)


def create_table(database_filepath, df, tbl_name):
    """
    Creates table in SQLite
    """
    conn = sqlite3.connect(database_filepath)

    print("Creating table in :" + database_filepath)
    df.to_sql(name=tbl_name, con=conn, if_exists='replace')

    conn.close()


def load_data(messages_filepath, categories_filepath):

    """
    Loads data from csv, and provides some processing e.g. splitting column into 36 other columns

    :param messages_filepath:
    :param categories_filepath:
    :return: df_messages, df_categories
    """

    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    # checking for nulls
    nulls = df_categories['categories'].isna().sum()
    print(nulls)

    col_split = df_categories['categories'].str.split('[;-]')
    rows = df_categories['categories'].str.split(';')
    matrix = []
    # count = 0
    for row in rows:
        row_arr = []
        for vals in row:
            num = vals[-1]
            # print(num)
            row_arr.append(num)
        matrix.append(row_arr)

    col_split_row = col_split[0]
    columns = []
    for col in col_split_row:
        if not col.isdigit():
            columns.append(col)
    dat = pd.DataFrame(matrix, columns=columns)
    df_categories = pd.concat([df_categories, dat], axis=1)
    print(df_categories.head())
    # print(map(lambda x: x[:-1], test))

    return df_messages, df_categories


def clean_data(df):
    """
    Merges two pandas dataframes to produce a combined dataframe

    :param df:
    :return: df
    :return: merged_df
    """

    merged_df = pd.concat([df[0], df[1]], axis=1)
    merged_df.drop(merged_df.columns[4], axis=1, inplace=True)
    merged_df = merged_df.replace('2', '0')
    cols = merged_df.iloc[:, 5:-1].columns
    # print(cols)
    for col in cols:
        print(col + ": " + str(merged_df[col].unique()))
        merged_df[col] = merged_df[col].astype(int)
    print(merged_df.iloc[:, 5:-1].dtypes)
    # print(merged_df.head())
    return df, merged_df


def save_data(df, df_merged, database_filepath):
    """
    Creates 3 tables; Categories, Messages, comb_table

    :param df:
    :param df_merged:
    :param database_filepath:
    :return:
    """
    # Categories table
    create_table(database_filepath, df[0], "Categories")
    # Messages table
    create_table(database_filepath, df[1], "Messages")
    # Concated table

    # print(merged_df.iloc[:,4])
    create_table(database_filepath, df_merged, "comb_table")


def main():
    """
    Main function
    :return:
    """


    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]# pylint: disable=unbalanced-tuple-unpacking

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    '
              f'CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df, df_merged = clean_data(df)

        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, df_merged, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
