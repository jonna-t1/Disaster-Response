import pickle
import sys
from sqlite3 import connect

import nltk
import numpy as np
import pandas as pd
# Sklearn
from sklearn import ensemble, neighbors, tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV TODO
# NLP
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


from Grid import Grid
nltk.download('stopwords')

"""
Trains the classifier against the data, loaded from the database
"""


def sampleSize(train_set, sample_size):
    """
    Allows the user to reduce the number of samples in the dataset

    :param train_set:
    :param sample_size:
    :return: rating1_sample
    """

    np.random.seed(123)  # this was used to produce the same every time
    rating_indices = train_set.index
    random_indices = np.random.choice(
        rating_indices, sample_size, replace=False)
    rating1_sample = train_set.loc[random_indices]
    return rating1_sample


def load_data(database_filepath):
    '''
    Load data and split into X matrix and y vector
    '''
    conn = connect(database_filepath)
    df = pd.read_sql('SELECT * from comb_table', conn)
    # print(df.iloc[:, 5:].columns)
    # for col in df.iloc[:, 5:].columns:
    #     df[col] = df[col].astype(int)
    # print(df["related"].unique())
    # dataframe[dataframe["column"] == value]
    df = sampleSize(df, 500)
    # print(df)
    X = df["message"].values
    cols = df.drop(["index", "message", "original",
                   "categories", "genre"], axis=1).columns

    df = df.drop(["index", "message", "original",
                 "categories", "genre"], axis=1)

    for col in cols:
        print(col + ": " + str(df[col].unique()))
        df[col] = df[col].astype(int)
    Y = df.values
    # print(Y["related"].unique())

    return X, Y, cols


def tokenize(text):
    """
    Tokenizes text data so it can be used by the model

    :param text:
    :return: words
    """
    import re
    # remove punctuation, special chars and digits
    regex = re.compile('[\'&*:#,\\.!?-_-\\$/()\\d+]')
    text = regex.sub('', text)
    words = word_tokenize(text)  # split text

    words = [w for w in words if w not in stopwords.words(
        "english")]  # remove stop words
    # remove words shorter than 3 chars
    i = 0
    for word in words:
        if len(word) <= 3:
            del words[i]
        i += 1
    return words


def build_model():
    """
    Selects a model and returns a pipeline of  CountVectorizer(), TfidfTransformer(),
    MultiOutputClassifier()

    :return: pipeline
    """

    rf = ensemble.RandomForestClassifier()
    # rnn = neighbors.RadiusNeighborsClassifier(radius=3.0)
    # knn = neighbors.KNeighborsClassifier()
    # en_xt = ensemble.ExtraTreesClassifier()
    # xt = tree.ExtraTreeClassifier()
    # dt = tree.DecisionTreeClassifier()
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(rf, n_jobs=-1))
    ])
    return pipeline


def evaluate_model(model, X_test, y_test, category_names, results):
    """
    Report the f1 score, precision and recall for each output category of the dataset.

    :param model:
    :param X_test:
    :param y_test:
    :param category_names:
    :param results:
    :return:
    """
    # print("SGD/LogReg Accuracy on new training set: {:.3f}".format(model.score(X_new, y_new)))
    print(
        "SGD/LogReg Accuracy on original test set: {model.score(X_test, y_test):.3f}")
    y_pred = results.predict(X_test)
    print('accuracy {accuracy_score(y_pred, y_test)}')
    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Saves the model

    :param model:
    :param model_filepath:
    :return:
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def get_input():
    """
    TODO
    Provides user input to choose avaiable multiclass multioutput classifiers
    :return:
    """

    prompt = 'Please input a number from 1-6 based on the multi-output multi-class ' \
             'classification model\n\n' \
             '1:        ensemble.RandomForestClassifier()\n' \
             '2:        neighbors.RadiusNeighborsClassifier()\n' \
             '3:        neighbors.KNeighborsClassifier()\n' \
             '4:        ensemble.ExtraTreesClassifier()\n' \
             '5:        tree.ExtraTreeClassifier()\n' \
             '6:        tree.DecisionTreeClassifier()\n\n'
    num_input = input(prompt)
    num_input = int(num_input)
    return num_input


def classify_ops(input, X_train, Y_train, X_test, Y_test, category_names):
    """
    Function that works with the Grid class

    :param input:
    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return:
    """

    model = Grid(X_train, Y_train, X_test, Y_test, category_names)
    match input:
        case 1:
            model.rforest_classifier()
        case 2:
            model.nr_NeighborsClassifier()
        case 3:
            model.nr_NeighborsClassifier()
        case 4:
            model.ensemble_extra_Trees()
        case 5:
            model.extra_treeClassifier()
        case 6:
            model.decision_tree()


def main():
    """
    Main method for the file
    :return:
    """

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        # input = get_input()
        # print("You picked " + str(input))
        # input = 2
        # print("Selecting option 2 - will be building other options later....")

        print('Loading data...\n    DATABASE: {database_filepath}')

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        results = model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names, results)
        # return
        print('Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl'
              'Or use the model filepath for an already loaded model'
              'train_classifier.py classifier.pkl'

              )


if __name__ == '__main__':
    main()
