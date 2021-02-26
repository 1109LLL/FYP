import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import pydotplus
import pickle
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn import preprocessing

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from bs4 import BeautifulSoup


def construct_feature_set(X_target, X_other1, X_other2):
    y_target = [0] * len(X_target)
    y_other1 = [1] * len(X_other1)
    y_other2 = [1] * len(X_other2)

    X = X_target+X_other1+X_other2
    y = y_target+y_other1+y_other2

    return X, y



def visualise_dataset(x_train, y_train, x_test, y_test):
    plt.figure(figsize=(6, 4))
    plt.scatter(x_train, y_train, color='red', label='Training set')
    plt.scatter(x_test, y_test, color='blue', label='Test set')
    plt.title('The data')
    plt.legend(loc='best')


def visualise_prediction(x_test,y_test,x_predict,y_predict):
    plt.figure(figsize=(6, 4))
    plt.scatter(x_predict, x_predict, color='red', label='Predicted')
    plt.scatter(x_test, y_test, color='blue', label='Test set')
    plt.title('The data')
    plt.legend(loc='best')


def cleaning(X):

    sentences = []
    for txt in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[txt]))
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Removing words with digits inside
        document = re.sub(r'\w*\d\w*', '', document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        document = word_tokenize(document)

        document = [stemmer.lemmatize(word, pos="n") for word in document if word not in stop_words]
        document = ' '.join(document)

        sentences.append(document)

    return sentences


def training(X, y, path_to_pickled_folder):
    # contruct features set
    vectorizer = CountVectorizer(max_features=200, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(X).toarray()
    feature_cols = vectorizer.get_feature_names()

    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()


    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    pickle_out = open(path_to_pickled_folder+"negative_model.pickle","wb")
    pickle.dump(clf, pickle_out)
    pickle_out.close()
    # pickle_in = open(path_to_pickled_folder+"specify model".pickle","rb")
    # clf = pickle.load(pickle_in)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print(metrics.classification_report(y_test,y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


    # dot_data = StringIO()
    export_graphviz(clf, out_file=path_to_pickled_folder+"negative_tree.dot",
                    filled=True, rounded=True,
                    special_characters=True,
                    feature_names = feature_cols,
                    class_names=['negative','other'])


def predictions(clf, path_to_pickled_folder):
    pickle_in = open(path_to_pickled_folder+"X_pos_sent.pickle","rb")
    X_pos_sent = pickle.load(pickle_in)

    pickle_in = open(path_to_pickled_folder+"X_neutral_sent.pickle","rb")
    X_neutral_sent = pickle.load(pickle_in)

    pickle_in = open(path_to_pickled_folder+"X_neg_sent.pickle","rb")
    X_neg_sent = pickle.load(pickle_in)
    X, y = construct_feature_set(X_neutral_sent, X_neg_sent, X_pos_sent)

    # contruct features set
    vectorizer = CountVectorizer(max_features=200, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(X).toarray()

    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print(X_test[0])

    # Model Accuracy, how often is the classifier correct?
    print("Mar-Apr::neutral model -> Nov-Dec")
    print(metrics.classification_report(y_test,y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


def starter(path_to_labelled_data, path_to_pickled_folder):

    # importing data
    candidate_sentences = pd.read_csv(path_to_labelled_data, encoding='ISO-8859-1')
    # pickle_out = open(path_to_pickled_folder+"dataset.pickle","wb")
    # pickle.dump(candidate_sentences, pickle_out)
    # pickle_out.close()
    # pickle_in = open(path_to_pickled_folder+"dataset.pickle","rb")
    # candidate_sentences = pickle.load(pickle_in)


    X_pos = candidate_sentences.loc[candidate_sentences['sentiment'] == "positive"]
    X_neutral = candidate_sentences.loc[candidate_sentences['sentiment'] == "neutral"]
    X_neg = candidate_sentences.loc[candidate_sentences['sentiment'] == "negative"]

    X_pos_sent = X_pos['sentence'].tolist()
    X_neutral_sent = X_neutral['sentence'].tolist()
    X_neg_sent = X_neg['sentence'].tolist()

    X_pos_sent = cleaning(X_pos_sent)
    # pickle_out = open(path_to_pickled_folder+"X_pos_sent.pickle","wb")
    # pickle.dump(X_pos_sent, pickle_out)
    # pickle_out.close()
    # pickle_in = open(path_to_pickled_folder+"X_pos_sent.pickle","rb")
    # X_pos_sent = pickle.load(pickle_in)

    X_neutral_sent = cleaning(X_neutral_sent)
    # pickle_out = open(path_to_pickled_folder+"X_neutral_sent.pickle","wb")
    # pickle.dump(X_neutral_sent, pickle_out)
    # pickle_out.close()
    # pickle_in = open(path_to_pickled_folder+"X_neutral_sent.pickle","rb")
    # X_neutral_sent = pickle.load(pickle_in)

    X_neg_sent = cleaning(X_neg_sent)
    # pickle_out = open(path_to_pickled_folder+"X_neg_sent.pickle","wb")
    # pickle.dump(X_neg_sent, pickle_out)
    # pickle_out.close()
    # pickle_in = open(path_to_pickled_folder+"X_neg_sent.pickle","rb")
    # X_neg_sent = pickle.load(pickle_in)


    print("POSITIVE training set = {}".format(len(X_pos_sent)))
    print("NEUTRAL training set = {}".format(len(X_neutral_sent)))
    print("NEGATIVE training set = {}".format(len(X_neg_sent)))
    X, y = construct_feature_set(X_neg_sent, X_pos_sent, X_neutral_sent)
    training(X,y, path_to_pickled_folder)

def visualise_accuracy():
    # positive
    months = ['Mar-Apr', 'May-Jun', 'Jul-Aug', 'Sep-Oct', 'Nov-Dec']
    pos_accuracy = [0.89, 0.83, 0.84, 0.84, 0.83]
    neg_accuracy = [0.94, 0.91, 0.92, 0.91, 0.91]
    neutral_accuracy = [0.81, 0.76, 0.78, 0.75, 0.77]
    plt.figure(figsize=(8, 4))
    plt.plot(months, pos_accuracy, color='red', label='positive')
    plt.plot(months, neg_accuracy, color='blue', label='negative')
    plt.plot(months, neutral_accuracy, color='orange', label='neutral')
    plt.ylabel('Accuracy')
    plt.title('Using positive classifer trained from Mar-Apr to predict positive sentiments')
    plt.legend(loc='best')
    plt.show()


########
# Main #
########


stemmer = WordNetLemmatizer()
stop_words = nltk.corpus.stopwords.words('english')

# training classifiers
######################
path_to_labelled_data = "../Mar-Apr_dataset/labelled_data.csv"
path_to_pickled_folder = "../Mar-Apr_pickled/"
starter(path_to_labelled_data, path_to_pickled_folder)

# path_to_labelled_data = "../May-Jun_dataset/labelled_data.csv"
# path_to_pickled_folder = "../May-Jun_pickled/"
# starter(path_to_labelled_data, path_to_pickled_folder)

# path_to_labelled_data = "../Jul-Aug_dataset/labelled_data.csv"
# path_to_pickled_folder = "../Jul-Aug_pickled/"
# starter(path_to_labelled_data, path_to_pickled_folder)

# path_to_labelled_data = "../Sep-Oct_dataset/labelled_data.csv"
# path_to_pickled_folder = "../Sep-Oct_pickled/"
# starter(path_to_labelled_data, path_to_pickled_folder)

# path_to_labelled_data = "../Nov-Dec_dataset/labelled_data.csv"
# path_to_pickled_folder = "../Nov-Dec_pickled/"
# starter(path_to_labelled_data, path_to_pickled_folder)




# Deterioration of classifiers
##############################
# path_to_pickled_folder = "../Mar-Apr_pickled/"
# pickle_in = open(path_to_pickled_folder+"neutral_model.pickle","rb")
# clf = pickle.load(pickle_in)


# path_to_target_pickled_folder = "../May-Jun_pickled/"
# predictions(clf, path_to_target_pickled_folder)

# path_to_target_pickled_folder = "../Jul-Aug_pickled/"
# predictions(clf, path_to_target_pickled_folder)

# path_to_target_pickled_folder = "../Sep-Oct_pickled/"
# predictions(clf, path_to_target_pickled_folder)

# path_to_target_pickled_folder = "../Nov-Dec_pickled/"
# predictions(clf, path_to_target_pickled_folder)

visualise_accuracy()