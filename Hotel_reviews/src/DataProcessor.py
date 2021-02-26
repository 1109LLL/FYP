import pandas as pd
import os
import re
import nltk
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import export_graphviz

from IPython.display import Image  

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from bs4 import BeautifulSoup

from pathlib import Path
from DataProvider import Getter
from Analyser import Predictor
from tqdm import tqdm


class FeatureReviewsExtractor:

    def positive_negative_reviews_on_feature(self,df,feature):
        pos_col = df["Positive_Review"]
        neg_col = df["Negative_Review"]

        positive = pos_col[pos_col.str.contains("(?i){}".format(feature))]
        negative = neg_col[neg_col.str.contains("(?i){}".format(feature))]
        return positive, negative

    def hotel_category_reviews(self,df,hotel_name,category):
        # df containing [hotel_name|neg_review|pos_review]
        rows_of_hotel = df.loc[df['Hotel_Name'].isin([hotel_name])]
        
        positive, negative = self.positive_negative_reviews_on_feature(rows_of_hotel,category)
        return positive, negative



class DataPreprocessing:
    def simplify(self,sent):
        # sent is tokenised parameter
        # use pos to find adj words
        
        treebankTagger = nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle')
        pos_tagged = treebankTagger.tag(sent)
        adj = []
        for i in range(len(sent)):
            if pos_tagged[i][1].startswith("JJ"):
                adj.append(pos_tagged[i][0])
        return adj

    def feature_matcher(self, sentence, feature):
        sents = re.findall(r'([A-Z][a-z|\s]+)', sentence)
        sent_matched_feature = [s for s in sents if feature in s]
        return sent_matched_feature

    def cleaning(self,X, feature):
        stemmer = WordNetLemmatizer()
        sentences = []

        more_stopwords = ['nothing','thing','without','really','minor','able','real','actual','previous','arrive','asmathic']
        stop_words = nltk.corpus.stopwords.words('english')
        stop_words.extend(more_stopwords)

        for txt in range(0, len(X)):
            raw_sent = str(X[txt])
            sent_matched_feature = self.feature_matcher(raw_sent, feature)

            for i in range(0, len(sent_matched_feature)):
                # Remove all the special characters
                document = re.sub(r'\W', ' ', sent_matched_feature[i])
                # remove all single characters
                document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
                # Remove single characters from the start
                document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
                # Substituting multiple spaces with single space
                document = re.sub(r'\s+', ' ', document, flags=re.I)
                # Removing words with digits inside
                document = re.sub(r'\w*\d\w*', '', document)
                # Converting to Lowercase
                document = document.lower()
                # Lemmatization
                document = word_tokenize(document)
                document = [stemmer.lemmatize(word, pos="n") for word in document if word not in stop_words]

                adjs = self.simplify(document)
                adjs = ' '.join(adjs)
                sentences.append(adjs)

        return sentences

    def clean_without_feature(self, X):
        stemmer = WordNetLemmatizer()
        sentences = []

        # more_stopwords = ['nothing','thing','without','really','minor','able','real','actual','previous','arrive','asmathic']
        stop_words = nltk.corpus.stopwords.words('english')
        # stop_words.extend(more_stopwords)

        for txt in range(0, len(X)):
            # Remove all the special characters
            document = re.sub(r'\W', ' ', str(X[txt]))
            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
            # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
            # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)
            # Removing words with digits inside
            document = re.sub(r'\w*\d\w*', '', document)
            # Converting to Lowercase
            document = document.lower()

            document = word_tokenize(document)
            cleaned_document = [stemmer.lemmatize(word, pos="n") for word in document if word not in stop_words]
            cleaned_document = ' '.join(cleaned_document)
            sentences.append(cleaned_document)
        
        return sentences

    def simple_cleaning(self, X, feature):
        stemmer = WordNetLemmatizer()
        sentences = []

        more_stopwords = ['nothing','thing','without','really','minor','able','real','actual','previous','arrive','asmathic']
        stop_words = nltk.corpus.stopwords.words('english')
        stop_words.extend(more_stopwords)

        for txt in range(0, len(X)):
            raw_sent = str(X[txt])
            sent_matched_feature = self.feature_matcher(raw_sent, feature)

            for i in range(0, len(sent_matched_feature)):
                # Remove all the special characters
                document = re.sub(r'\W', ' ', sent_matched_feature[i])
                # remove all single characters
                document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
                # Remove single characters from the start
                document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
                # Substituting multiple spaces with single space
                document = re.sub(r'\s+', ' ', document, flags=re.I)
                # Removing words with digits inside
                document = re.sub(r'\w*\d\w*', '', document)
                # Converting to Lowercase
                document = document.lower()
                
                document = word_tokenize(document)
                cleaned_document = [stemmer.lemmatize(word, pos="n") for word in document if word not in stop_words]
                cleaned_document = ' '.join(cleaned_document)
                sentences.append(cleaned_document)

        return sentences

    def cleaning_for_preprocessing3(self, X, feature):
        stemmer = WordNetLemmatizer()
        sentences = []

        more_stopwords = ['nothing','thing','without','really','minor','able','real','actual','previous','arrive','asmathic']
        stop_words = nltk.corpus.stopwords.words('english')
        stop_words.extend(more_stopwords)

        for txt in range(0, len(X)):
            raw_sent = str(X[txt])
            sent_matched_feature = self.feature_matcher(raw_sent, feature)
            for i in range(0, len(sent_matched_feature)):
                document = re.sub(r'\W', ' ', sent_matched_feature[i])
                # remove all single characters
                document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
                # Removing words with digits inside
                document = re.sub(r'\w*\d\w*', '', document)
                # Substituting multiple spaces with single space
                document = re.sub(r'\s+', ' ', document, flags=re.I)
                document = document.lower()

                document = word_tokenize(document)
                cleaned_document = [stemmer.lemmatize(word, pos="n") for word in document if word not in stop_words]
                try:
                    index_ = cleaned_document.index(feature)
                except:
                    continue
                word_1 = ""
                word_2 = ""
                if index_ > 0:
                    word_1 = cleaned_document[index_-1]
                if index_ < len(cleaned_document)-1:
                    word_2 = cleaned_document[index_+1]
                words = [word_1, word_2]
                cleaned_document = ' '.join(words)
                sentences.append(cleaned_document)
        return sentences


    def construct_feature_set(self, X_target, y_other):
        y_target_label = [0] * len(X_target)
        y_other_label = [1] * len(y_other)

        X = X_target+y_other
        y = y_target_label+y_other_label
        # print("X_target size = {}".format(len(X_target)))
        # print("y_other size = {}".format(len(y_other)))
        # print("X size = {}".format(len(X)))
        # print("y size = {}".format(len(y)))
        return X, y
    
    def preprocessing(self, X_, y_, feature):
        X_ = X_.values.tolist()
        y_ = y_.values.tolist()
        
        X_sentence = self.cleaning(X_, feature)
        y_sentence = self.cleaning(y_, feature)
        return X_sentence, y_sentence

    def preprocessing_2(self, X_, y_, feature):
        X_ = X_.values.tolist()
        y_ = y_.values.tolist()
        
        X_sentence = self.simple_cleaning(X_, feature)
        y_sentence = self.simple_cleaning(y_, feature)
        return X_sentence, y_sentence

    def preprocessing_3(self, X_, y_, feature):
        X_ = X_.values.tolist()
        y_ = y_.values.tolist()

        X_sentence = self.cleaning_for_preprocessing3(X_, feature)
        y_sentence = self.cleaning_for_preprocessing3(y_, feature)
        return X_sentence, y_sentence


class Categorical:
    def labelling(self):
        getter = Getter()
        categories = getter.getCategories()
        hotel_list = getter.getHotelList()
        cols = ['Hotel_Name','Negative_Review','Positive_Review']
        df = getter.getSelectedCols(cols)
        
        vectorizer, clf = getter.getPosNegReviewsClf()
        predictor = Predictor()
        extract = FeatureReviewsExtractor()
        
        print("Number of unique hotels = {}".format(len(hotel_list)))
        # loop through each category/label, find matching hotels
        for cat in categories:
            folder_name = cat
            record = {'Hotel_Name':[]}
            cat = cat.replace('_', ' ')

            for hotel in tqdm(hotel_list):
                positive, negative = extract.hotel_category_reviews(df, hotel, cat)
                reviews = positive.append(negative)
                if reviews.empty:
                    continue

                reviews_list = reviews.values.tolist()
                result = predictor.input_predictor(reviews_list, vectorizer, clf)

                num_pos = np.count_nonzero(result==0)
                num_neg = np.count_nonzero(result==1)
                # print(num_pos)
                # print(num_neg)
                # print("{} percent ".format(num_pos/(num_pos+num_neg)))
                
                # add satisfactory hotel names to list
                # Only hotels with reviews that are 95% or more positive will be added
                if (num_pos/(num_pos+num_neg)) > 0.95:
                    record['Hotel_Name'].append(hotel)

                # if np.count_nonzero(result==0) > np.count_nonzero(result==1):
                    # record['Hotel_Name'].append(hotel)
                
            dataframe = pd.DataFrame(record, columns = ['Hotel_Name'])
            path = '../labelled_hotels/{}.csv'.format(folder_name)
            dataframe.to_csv(path, index=False)
            print("[{}] category = {}".format(folder_name,len(record['Hotel_Name'])))


