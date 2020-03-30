#!/usr/bin/env python

# import all necessary libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


# load data and merge function
def load_data(messages_filepath, categories_filepath):
    """
    Function to load data and merge both files as one

    input:
    - messages_filepath : filepath to the messages csv file
    - categories_filepath : filepath to the categories csv

    returns:
    - a merged pandas dataframe of both messages and categories csv file

    """

    df_messages = pd.read_csv(messages_filepath)
    # print(df_messages.head())
    df_categories = pd.read_csv(categories_filepath)
    # print(df_categories.head())

    return pd.merge(df_messages, df_categories, on="id")


# clean dataframe function
def clean_data(df):
    """
    Function to:
    - clean some messy columns
    - drop duplicate data
    - change data type as necessary

    input:
    - df : pandas dataframe

    returns:
    - cleaned dataframe

    """
    # split categories column, expand, change column names and extract only relevant part of name
    categories = df.categories.str.split(";", expand=True)
    # print(categories.head())
    row = categories.loc[0]
    categories.columns = row.apply(lambda x: x[:-2])

    # extract number from column and make column an integer column
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:].astype(int)
    # print(categories.head())

    # merge categories dataframe with df dataframe
    df = pd.concat([df.drop("categories", axis=1), categories], axis=1)
    # print(df.head())

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


# save data function
def save_data(df, database_filename):
    """
    Function to save data to sql database

    input:
    - df : pandas dataframe
    - database_filename : database name

    returns:
    - null

    """
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql("disaster_data_cleaned", con=engine, if_exists="replace", index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        # print(df.shape)
        # print(df.head())

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
