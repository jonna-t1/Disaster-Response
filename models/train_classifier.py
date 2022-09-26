import pickle
import sys
from sqlite3 import connect

import joblib
import nltk
import numpy as np
import pandas as pd
from sklearn import ensemble, neighbors, tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from Grid import Grid

from sklearn.model_selection import train_test_split, GridSearchCV

## NLP
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# nltk.download()
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

nltk.download('stopwords')

def produceSample(train_set, sample_size):
    np.random.seed(123)     ## this was used to produce the same every time
    rating_indices = train_set.index
    random_indices = np.random.choice(rating_indices, sample_size, replace=False)
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
    df = produceSample(df, 500)
    # print(df)
    X = df["message"].values
    cols = df.drop(["index", "message", "original", "categories", "genre"], axis=1).columns

    df = df.drop(["index", "message", "original", "categories", "genre"], axis=1)
    df = df.replace('2','0')

    for col in cols:
        print(col + ": "+ str(df[col].unique()))
        df[col] = df[col].astype(int)
    Y = df.values
    # print(Y["related"].unique())

    return X, Y, cols


def tokenize(text):
    import re
    regex = re.compile('[\'&*:#,\.!?-_-\$/()\d+]') #remove punctuation, special chars and digits
    text = regex.sub('', text)
    words = word_tokenize(text) #split text

    words = [w for w in words if w not in stopwords.words("english")] #remove stop words
    #remove words shorter than 3 chars
    i = 0
    for word in words:
        if (len(word) <= 3):
            del words[i]
        i+=1
    return words

def build_model():
    rf = ensemble.RandomForestClassifier()
    rnn = neighbors.RadiusNeighborsClassifier(radius=3.0)
    knn = neighbors.KNeighborsClassifier()
    en_xt = ensemble.ExtraTreesClassifier()
    xt = tree.ExtraTreeClassifier()
    dt = tree.DecisionTreeClassifier()
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(rf, n_jobs=-1))
    ])
    return pipeline

# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through
# the columns and calling sklearn's classification_report on each.
def evaluate_model(model, X_test, y_test, category_names, results):
    # print("SGD/LogReg Accuracy on new training set: {:.3f}".format(model.score(X_new, y_new)))
    print("SGD/LogReg Accuracy on original test set: {:.3f}".format(model.score(X_test, y_test)))
    y_pred = results.predict(X_test)
    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred, target_names=category_names))
    pass


def save_model(model, model_filepath):
    #save the model
    pickle.dump(model, open(model_filepath, 'wb'))

def get_input():
    prompt = 'Please input a number from 1-6 based on the multi-output multi-class classification model\n\n' \
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
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        # input = get_input()
        # print("You picked " + str(input))
        input = 2
        print("Selecting option 2 - will be building other options later....")

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # classify_ops(input, X_train, Y_train, X_test, Y_test, category_names)
        # return
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        results = model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names, results)
        # return
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    # if len(sys.argv) == 4:
    #     import timeit
    #     start = timeit.default_timer()
    #     database_filepath, model_filepath, num = sys.argv[1:]
    #     X, Y, category_names = load_data(database_filepath)
    #     # print(X.head())
    #     my_tags = category_names.tolist()
    #
    #     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    #     if
    #     grid = Grid(2,3)
    #     print(grid)
    #     # print(X_train.shape)
    #     # print(X_test.shape)
    #     # print(Y_train[0])
    #     # print("\n\n\n")
    #     # print(Y_train)
    #     return
    #     # load the model from disk
    #     loaded_model = joblib.load(model_filepath)
    #     stop1 = timeit.default_timer()
    #     print('Time: ', stop1 - start)
    #
    #     loaded_model.fit(X_train,Y_train)
    #     Y_pred = loaded_model.predict(X_test)
    #     print(Y_pred.shape)
    #     stop2 = timeit.default_timer()
    #     print('Time: ', stop2 - start)

    #
    #     # print(result)
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl'\
              'Or use the model filepath for an already loaded model'\
              'train_classifier.py classifier.pkl'

              )


if __name__ == '__main__':
    main()