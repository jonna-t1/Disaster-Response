import pickle
import sys
from sqlite3 import connect

import joblib
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.model_selection import train_test_split

## NLP
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# nltk.download()
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

nltk.download('stopwords')


def load_data(database_filepath):
    '''
    Load data and split into X matrix and y vector
    '''
    conn = connect(database_filepath)
    df = pd.read_sql('SELECT * from comb_table', conn)
    # print(df.head())
    X = df["message"].values
    Y = df.drop(["index", "message", "original", "categories", "genre"], axis=1).values
    cols = df.drop(["index", "message", "original", "categories", "genre"], axis=1).columns
    # print(Y.head())
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
    knn = KNeighborsClassifier(n_neighbors=36)
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        # ('clf', MultiOutputClassifier(knn, n_jobs=-1))
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    # print("SGD/LogReg Accuracy on new training set: {:.3f}".format(model.score(X_new, y_new)))
    print("SGD/LogReg Accuracy on original test set: {:.3f}".format(model.score(X_test, Y_test)))
    pass


def save_model(model, model_filepath):
    #save the model
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        # X_train_tok = []
        # for i in X_train:
        #     X_train_tok.append(tokenize(i))
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    if len(sys.argv) == 4:
        import timeit
        start = timeit.default_timer()
        database_filepath, model_filepath, num = sys.argv[1:]
        X, Y, category_names = load_data(database_filepath)
        my_tags = category_names.tolist()

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print(X_train.shape)
        print(X_test.shape)
        print(Y_train[0])
        print("\n\n\n")
        print(Y_test.shape)
        # return
        # load the model from disk
        loaded_model = joblib.load(model_filepath)
        stop1 = timeit.default_timer()
        print('Time: ', stop1 - start)

        loaded_model.fit(X_train,Y_train)
        Y_pred = loaded_model.predict(X_test)
        print(Y_pred.shape)
        stop2 = timeit.default_timer()
        print('Time: ', stop2 - start)
        print('accuracy %s' % accuracy_score(Y_pred, Y_test))
        print(classification_report(Y_test, Y_pred, target_names=my_tags))

        # print(result)
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