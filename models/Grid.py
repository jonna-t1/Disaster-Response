from nltk import word_tokenize
from nltk.corpus import stopwords
from numpy import arange
from sklearn import ensemble, neighbors, tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


def tokenise(text):
    import re
    regex = re.compile('[\'&*:#,\.!?-_-\$/()\d+]')  # remove punctuation, special chars and digits
    text = regex.sub('', text)
    words = word_tokenize(text)  # split text

    words = [w for w in words if w not in stopwords.words("english")]  # remove stop words
    # remove words shorter than 3 chars
    i = 0
    for word in words:
        if (len(word) <= 3):
            del words[i]
        i += 1
    return words

class Grid:
    def __new__(cls, *args, **kwargs):
        # print("1. Create a new instance of Grid.")
        return super().__new__(cls)

    def __init__(self, trainX, trainY, X_test, Y_test, category_names):
        # print("2. Initialize the new instance of Grid.")
        self.X = trainX
        self.y = trainY
        self.X_test = X_test
        self.y_test = Y_test
        self.my_tags = category_names

    def __repr__(self) -> str:
        return f"{type(self).__name__}(x={self.X}, y={self.y})"

    def eval(self,search):
        # perform the search
        print(self.X.shape)
        print(self.y.shape)
        results = search.fit(self.X, self.y)
        # print(results.cv_)
        # summarize
        print(len(self.X_test))
        y_pred = results.predict(self.X_test)
        print('accuracy %s' % accuracy_score(y_pred, self.y_test))
        print(classification_report(self.y_test, y_pred, target_names=self.my_tags))

    def createPipeline(self, classifier):
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', classifier)
        ])
        return pipeline

## start of the models
    def rforest_classifier(self):
        ensemble.RandomForestClassifier()
        print("rforest")

    def nr_NeighborsClassifier(self):
        model = neighbors.RadiusNeighborsClassifier(radius=3.0)
        print("nr_NeighborsClassifier")
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # define grid
        grid = dict()
        grid['model__radius'] = arange(0.8, 1.5, 0.01)
        pipeline = self.createPipeline(model)
        print(sorted(pipeline.get_params().keys()))
        search = GridSearchCV(pipeline, grid, scoring='accuracy', cv=5, n_jobs=-1)
        self.eval(search)
        # return search

    def rnClassifier(self):
        model = neighbors.KNeighborsClassifier()
        print("knnClassifier")
        return model

    def ensemble_extra_Trees(self):
        model = ensemble.ExtraTreesClassifier()
        print("extra trees")
        return model

    def extra_treeClassifier(self):
        model = tree.ExtraTreeClassifier()
        print("extra_tree Classifier")
        return model

    def decision_tree(self):
        model = tree.DecisionTreeClassifier()
        print("Decision tree")
        return model