import pandas as pd
import re
import nltk

from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO  
from IPython.display import Image  

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from bs4 import BeautifulSoup

import datetime
import pickle

class Trainer:

    def train(self, X, y, parameters):
        # contruct features set
        vectorizer = CountVectorizer(max_features=10, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
        X = vectorizer.fit_transform(X)
        print("Printing the vocabulary below")
        feature_cols = vectorizer.get_feature_names()
        print(feature_cols)
        X = X.toarray()
        print(X)
        

        tfidfconverter = TfidfTransformer()
        X = tfidfconverter.fit_transform(X).toarray()

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=8)
        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

         #Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        print("result of {} clf".format(parameters[1]))
        print(metrics.classification_report(y_test,y_pred))
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        # dot_data = StringIO()
        export_graphviz(clf, out_file="../{}/{}.dot".format(parameters[0], parameters[1]),
                        filled=True, rounded=True,
                        special_characters=True,
                        feature_names = feature_cols,
                        class_names=['positive','nagative'])
        return clf

    def tf_idf(self, reviews):
        try:
            cv = CountVectorizer(max_features=10, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
            word_count_vector = cv.fit_transform(reviews)

            tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True) 
            tfidf_transformer.fit(word_count_vector)

            # print idf values 
            df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"]) 
            
            # sort ascending 
            # sorted_values = df_idf.sort_values(by=['idf_weights'],ascending=False)
            # print(sorted_values[0:20])
            # count matrix 
            count_vector = cv.transform(reviews) 
            
            # tf-idf scores 
            tf_idf_vector = tfidf_transformer.transform(count_vector)

            feature_names = cv.get_feature_names()
            
            #get tfidf vector for first document 
            first_document_vector = tf_idf_vector[0] 
            
            #print the scores 
            df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"]) 
            print(df.sort_values(by=["tfidf"],ascending=False)[0:10])
            return feature_names[:5]
        except:
            return []
        
    def pos_neg_clf(self, X, y):
        print("##########################################")
        now = datetime.datetime.now()
        print("{} : start training".format(now.strftime("%Y-%m-%d %H:%M:%S")))

        vectorizer = CountVectorizer(max_features=200)
        # the number of times a particular feature words appears
        X = vectorizer.fit_transform(X).toarray()

        # Number of words features here = total number of unique words in your dataset
        feature_cols = vectorizer.get_feature_names()

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        now = datetime.datetime.now()
        print("{} : finished training".format(now.strftime("%Y-%m-%d %H:%M:%S")))
        print("##########################################")

        pickle_out = open("../pickled_clfs/pos_neg_review_vectorizer.pickle","wb")
        pickle.dump(vectorizer, pickle_out)
        pickle_out.close()

        pickle_out = open("../pickled_clfs/pos_neg_review_clf.pickle","wb")
        pickle.dump(clf, pickle_out)
        pickle_out.close()

        print("Vectorizer and clf stored successfully!")
        print("##########################################")

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        print(metrics.classification_report(y_test,y_pred))
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        export_graphviz(clf, out_file="../{}/{}.dot".format("feature_clfs_tree", "pos_neg_review"),
                        filled=True, rounded=True,
                        special_characters=True,
                        feature_names = feature_cols,
                        class_names=['positive','nagative'])

        return vectorizer, clf

class Predictor:
    def input_predictor(self,input_: list, vectorizer, clf) -> list:
        input_ = vectorizer.transform(input_).toarray()
        result = clf.predict(input_)
        return result
