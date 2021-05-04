import pandas as pd
import re
import nltk
import datetime
import pickle

from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ML algorithms
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

# Data preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO  

from bs4 import BeautifulSoup
from collections import Counter
from IPython.display import Image  

from DataProvider import Getter
from pandas.core.common import flatten

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

    def create_vectorizer(self, X, y, max_features, min_df, info):
        stop_words = nltk.corpus.stopwords.words('english')
        stop_words = set(stopwords.words('english')) - set(['no'])

        vectorizer = CountVectorizer(max_features=max_features, stop_words=stop_words)

        # the number of times a particular feature words appears
        X_transformed = vectorizer.fit_transform(X).toarray()

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=1)

        # bag_of_words_package = [X_train, X_test, y_train, y_test, vectorizer, X_transformed]
        # pickle_out = open("../pickled_clfs/{}.pickle".format(info), "wb")
        # pickle.dump(bag_of_words_package, pickle_out)
        # pickle_out.close()

        return X_train, X_test, y_train, y_test, vectorizer, X_transformed

    def sentiment_clf(self, X_train, X_test, y_train, y_test, feature_cols):
        print("------------------------------------------")
        now = datetime.datetime.now()
        print("{} : start training".format(now.strftime("%Y-%m-%d %H:%M:%S")))

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier()
        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        now = datetime.datetime.now()
        print("{} : finished training".format(now.strftime("%Y-%m-%d %H:%M:%S")))
        print("------------------------------------------")

        pickle_out = open("../pickled_clfs/default_pos_neg_review_clf.pickle","wb")
        pickle.dump(clf, pickle_out)
        pickle_out.close()

        print("Clf stored successfully!")
        print("------------------------------------------")

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        print(metrics.classification_report(y_test,y_pred))
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        export_graphviz(clf, out_file="../{}/{}.dot".format("default_clfs_tree", "pos_neg_review"),
                        filled=True, rounded=True,
                        special_characters=True,
                        feature_names = feature_cols,
                        class_names=['positive','nagative'])
        print("------------------------------------------")



    def form_training_set_for_sentiment_clf_decision_tree(self):
        getter = Getter()
        hotel_list = getter.getHotelList()

        pos_val = []
        neg_val = []
        # Extract the positive and negative dictionaries (noun and adj) from each hotel
        # obtain a list of only the values
        for hotel in hotel_list.values:
            hotel_name = hotel[0]

            pos_dict = getter.get_pos_noun_adj_dict_of_hotel(hotel_name)
            pos_values_list = list(pos_dict.values())
            for value in pos_values_list:
                if len(value) == 1:
                    pos_val.append(value[0])
                else:
                    pos_val.append(' '.join(value))


            neg_dict = getter.get_neg_noun_adj_dict_of_hotel(hotel_name)
            neg_values_list = list(neg_dict.values())
            for value_ in neg_values_list:
                if len(value_) == 1:
                    neg_val.append(value_[0])
                else:
                    neg_val.append(' '.join(value_))

        y_pos_label = [1] * len(pos_val)
        y_neg_label = [0] * len(neg_val)    
        
        X = pos_val + neg_val
        y = y_pos_label + y_neg_label

        return X, y

    def fit_training_set_using_tfidf_vectorizer(self):
        print("------------------------------------------")
        now = datetime.datetime.now()
        print("{} : start fitting tfidf".format(now.strftime("%Y-%m-%d %H:%M:%S")))

        getter = Getter()
        X, y = getter.get_training_set_for_sentiment_clf_decision_tree()
        
        tfIdfVectorizer = TfidfVectorizer(use_idf=True)
        X_tfidf = tfIdfVectorizer.fit_transform(X)

        now = datetime.datetime.now()
        print("{} : finished fit_transform".format(now.strftime("%Y-%m-%d %H:%M:%S")))
        print("------------------------------------------")

        return X_tfidf, tfIdfVectorizer


    def train_sentiment_clf_decision_tree(self):
        print("------------------------------------------")
        now = datetime.datetime.now()
        print("{} : start training".format(now.strftime("%Y-%m-%d %H:%M:%S")))

        getter = Getter()
        X_tfidf, y, tfIdfVectorizer = getter.get_X_tfidf_y_training_set_for_sentiment_clf_decision_tree()

        # df = pd.DataFrame(tfidf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
        # df = df.sort_values('TF-IDF', ascending=False)
        # print (df.head(25))


        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(criterion="gini", max_depth=1000)
        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        now = datetime.datetime.now()
        print("{} : finished training".format(now.strftime("%Y-%m-%d %H:%M:%S")))
        print("------------------------------------------")

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Model Accuracy
        print(metrics.classification_report(y_test,y_pred))
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        
        export_graphviz(clf, out_file="../generated_files/sentiment_clf_DecisionTreeClassifier/decision_tree.dot",
                        filled=True, rounded=True,
                        special_characters=True,
                        feature_names = tfIdfVectorizer.get_feature_names(),
                        class_names=['negative','positive'])
        print("Tree depth = {}".format(clf.get_depth()))
        print("Number of parameters in the model: {}".format(len(clf.get_params)))
        print("------------------------------------------")

        return clf

class Predictor:
    def input_predictor(self,input_: list, vectorizer, clf) -> list:
        input_ = vectorizer.transform(input_).toarray()
        label = clf.predict(input_)
        confidence = clf.predict_proba(input_)
        result = {'label':label, 'confidence':confidence}
        return result

class FeatureExtractor:
    def best_features_term_frequency_approach(self, hotel_name, pos_dict):
        best_features = {}
        # Find 5 features that are talked about the most.
        for k in sorted(pos_dict, key=lambda k: len(pos_dict[k]), reverse=True):
            # print(str(k)+":"+str(len(pos_dict[k]))+" = ",end="")
            # print(pos_dict[k])

            # Find the top 3 most used words that describes this feature
            values = Counter(pos_dict[k])
            sorted_values = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))

            best_three = {i: sorted_values[i] for i in list(sorted_values)[:3]}
            best_features[k] = best_three

            if len(best_features)==5:
                break
        # print(best_features)
        return best_features
