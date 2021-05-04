import nltk
import pickle
import regex as re
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
from sklearn.preprocessing import LabelEncoder
import numpy as np

from DataProcessor import FeatureReviewsExtractor, DataPreprocessing, TextAnalysis, Categorical
from DataProvider import Getter

from sklearn.model_selection import GridSearchCV

from Analyser import Trainer, FeatureExtractor, Predictor
from dtreeviz.trees import dtreeviz
from matplotlib import pyplot as plt

from spellchecker import SpellChecker

# words = ['minor','sandwich', 'able','private']
# treebankTagger = nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle')
# pos_tagged = treebankTagger.tag(words)
# print(pos_tagged)

# string_ = " No real complaints the hotel was great great location surroundings rooms amenities and service Two recommendations however firstly the staff upon check in are very confusing regarding deposit payments and the staff offer you upon checkout to refund your original payment and you can make a new one Bit confusing Secondly the on site restaurant is a bit lacking very well thought out and excellent quality food for anyone of a vegetarian or vegan background but even a wrap or toasted sandwich option would be great Aside from those minor minor things fantastic spot and will be back when i return to Amsterdam "
# sent = re.findall(r'([A-Z][a-z|\s]+)',string_)
# match = [s for s in sent if "room" in s]
# print(sent)
# print(match)



def predict(X, y):
    vectorizer = CountVectorizer(max_features=200)
    # the number of times a particular feature words appears
    X = vectorizer.fit_transform(X).toarray()
    print(X)
    print("length = {}".format(len(X)))
    print(X[0])

    inverted = vectorizer.inverse_transform(X[0])
    print(inverted)

    # Number of words features here = total number of unique words in your dataset
    feature_cols = vectorizer.get_feature_names()
    print(feature_cols)
    print("number of features = {}".format(len(feature_cols)))
    
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=8)
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print(metrics.classification_report(y_test,y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    # Predict new data:
    print("Predicting new data ....")
    sent = [" Rooms were stunningly decorated and really spacious in the top of the building Pictures are of room 300 The true beauty of the building has been kept but modernised brilliantly Also the bath was lovely and big and inviting Great more for couples Restaurant menu was a bit pricey but there were loads of little eatery places nearby within walking distance and the tram stop into the centre was about a 6 minute walk away and only about 3 or 4 stops from the centre of Amsterdam Would recommend this hotel to anyone it s unbelievably well priced too"]

    sent_cv = vectorizer.transform(sent).toarray()
    result = clf.predict(sent_cv)
    print("The result = {}".format(result))

def oldtest():

    pickle_in = open("../pickled_files/reviews.pickle","rb")
    df = pickle.load(pickle_in)
    extractor = FeatureReviewsExtractor()
    positive_review, negative_review = extractor.positive_negative_reviews_on_feature(df, "room")


    # # Clean reviews
    # processor = DataPreprocessing()
    # cleaned_pos_reviews, cleaned_neg_reviews = processor.preprocessing_2(positive_review, negative_review, "room")

    # # # Construct date set for training
    # X, y = processor.construct_feature_set(cleaned_pos_reviews, cleaned_neg_reviews)
    # print(X[:5])
    # print(y[:5])

    test_string = ["This room is absolutely horrible, not big enough, and it smells"]

    pickle_in = open("../pickled_clfs/{}.pickle".format("room"),"rb")
    clf = pickle.load(pickle_in)

    predict(clf,test_string)

def newtest():
    pickle_in = open("../pickled_files/reviews.pickle","rb")
    df = pickle.load(pickle_in)

    pos_col = df["Positive_Review"].head(10000)
    neg_col = df["Negative_Review"].head(10000)

    # Clean reviews
    processor = DataPreprocessing()
    cleaned_pos_reviews = processor.clean_without_feature(pos_col)
    cleaned_neg_reviews = processor.clean_without_feature(neg_col)

    # Construct data set for training
    X, y = processor.construct_feature_set(cleaned_pos_reviews, cleaned_neg_reviews)

    predict(X, y)



def create_document_term_matrix(x, vectorizer):
    return pd.DataFrame(x, columns=vectorizer.get_feature_names())

getter = Getter()
clf, tfIdfVectorizer = getter.get_sentiment_decision_tree_clf()

print("Tree depth = {}".format(clf.get_depth()))
print("Number of parameters in the model: {}".format(clf.get_params()))

# processor = TextAnalysis()
# sent = " Rooms are nice but for elderly a bit difficult as most rooms are two story with narrow steps So ask for single level Inside the rooms are very very basic just tea coffee and boiler and no bar empty fridge "
# a = " My room was dirty and I was afraid to walk barefoot on the floor which looked as if it was not cleaned in weeks White furniture which looked nice in pictures was dirty too and the door looked like it was attacked by an angry dog My shower drain was clogged and the staff did not respond to my request to clean it On a day with heavy rainfall a pretty common occurrence in Amsterdam the roof in my room was leaking luckily not on the bed you could also see signs of earlier water damage I also saw insects running on the floor Overall the second floor of the property looked dirty and badly kept On top of all of this a repairman who came to fix something in a room next door at midnight was very noisy as were many of the guests I understand the challenges of running a hotel in an old building but this negligence is inconsistent with prices demanded by the hotel On the last night after I complained about water damage the night shift manager offered to move me to a different room but that offer came pretty late around midnight when I was already in bed and ready to sleep "
# c = " The room was big enough and the bed is good "
# aa = processor.start(a)

# lexical_result = processor.lexical_analysis(c)
# chunks = processor.syntactic_analysis_for_negatives(lexical_result)
# chunks_dict = processor.noun_adj_dict(chunks)
# print(chunks_dict)



# msg = ["hi man how are you", "i just wanna sleep"]

# vec = CountVectorizer()
# x = vec.fit_transform(msg).toarray()
# df = create_document_term_matrix(x,vec)
# print(df)

# clf = DecisionTreeClassifier()

# X_path = "../pickled_files/X_training_set.pickle"
# y_path = "../pickled_files/y_training_set_label.pickle"
# pickle_in = open(X_path, "rb")
# X = pickle.load(pickle_in)
# pickle_in = open(y_path, "rb")
# y = pickle.load(pickle_in)

# print(X[0])
# print(X[1])
# print(X[2])
# print(len(X))
# print(y[0])
# print(y[1])
# print(len(y))

# newtest()
# cat = Categorical()

# cat.labelling()
# print(hotel)
# print(positive)
# print(len(positive))
# print(negative)
# print(len(negative))




####################
# trainer = Trainer()
    
# X_path = "../pickled_files/X_training_set.pickle"
# y_path = "../pickled_files/y_training_set_label.pickle"
# pickle_in = open(X_path, "rb")
# X = pickle.load(pickle_in)
# pickle_in = open(y_path, "rb")
# y = pickle.load(pickle_in)

# max_feature = 50
# info = "50_feature_vector"
# min_df = None
# X_train, X_test, y_train, y_test, vectorizer, X_transformed = trainer.create_vectorizer(X, y, max_feature, min_df, info)

# getter = Getter()
# vectorizer, clf = getter.getPosNegReviewsClf()
# fig = plt.figure(figsize=(25,20))

# viz = dtreeviz(clf, X_train, y_train,
#                 target_name="target",
#                 feature_names=vectorizer.get_feature_names(),
#                 class_names=["positive","nagative"])

# fig.savefig("decistion_tree.png")
# ####################