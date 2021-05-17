import pandas as pd
import os
import pickle
from tqdm import tqdm
import re
import glob
from DataProcessor import FeatureReviewsExtractor, DataPreprocessing
from Analyser import Trainer
from DataProvider import Getter, FileGenerator
from DataProcessor import Categorical
from DataProcessor import TextAnalysis
import datetime 


def feature_extraction(df, feature):
    # Extract reviews that mentions the feature
    extractor = FeatureReviewsExtractor()
    positive_review, negative_review = extractor.positive_negative_reviews_on_feature(df, feature)

    # Clean reviews
    processor = DataPreprocessing()
    cleaned_pos_reviews, cleaned_neg_reviews = processor.preprocessing_3(positive_review, negative_review, feature)
    # return cleaned_pos_reviews, cleaned_neg_reviews

    # Analyse data
    analyser = Trainer()
    pos_features = analyser.tf_idf(cleaned_pos_reviews)
    neg_features = analyser.tf_idf(cleaned_neg_reviews)
    return pos_features, neg_features



def simple_recommend():
    pickle_in = open("../pickled_files/reviews.pickle","rb")
    df = pickle.load(pickle_in)

    hotel_list = df.Hotel_Name.unique()
    sub_list = hotel_list[:10]
    
    
    while True:
        print(sub_list)
        hotel_entered = input("Please choose a hotel for viewing: ")
        while hotel_entered not in sub_list:
            hotel_entered = input("Please choose a valid hotel for viewing: ")
        hotel_info = df.loc[df['Hotel_Name'] == hotel_entered]
        print("\n")

        getter = Getter()
        features = getter.getFeatureList()
        print("Here is a list of features that may be of interest")
        for i in range(0, len(features)):
            print("{}. {}".format(i+1, features[i]))
        feature_selected = input("Please enter the feature that you would like to see: ")
        while feature_selected not in features:
            feature_selected = input("please choose a valid feature: ")

        pos_adj, neg_adj = feature_extraction(hotel_info, feature_selected)
        print("Below is what other people thinks about this feature:")
        print("positives: {}".format(pos_adj))
        print("negatives: {}".format(neg_adj))
        print("\n")
        decision = input("Would you like to see another hotel? [Y/n] : ")
        if decision == "N" or decision == "n":
            break
        print("\n")

def advanced_recommend():
    pickle_in = open("../pickled_files/reviews.pickle","rb")
    df = pickle.load(pickle_in)

    hotel_list = df.Hotel_Name.unique()
    sub_list = hotel_list[:10]
    getter = Getter()
    categories = getter.getCategories()

    while True:
        print(categories)
        category_entered = input("Please choose categories: ")
        while category_entered not in categories:
            category_entered = input("Please choose a valid category for recommendations: ")
        
        dataframe = []
        for hotel in sub_list:
            dataframe.append(df.loc[df['Hotel_Name'] == hotel])
        hotel_info = pd.concat(dataframe,ignore_index=True)
        
        # pos_adj, neg_adj = feature_extraction(hotel_info, feature_entered)

        # selected_feature_pos = pos_adj[:3]
        # print("Please see recommended hotels with the following descriptions:")

def generate_files():
    # Generate the files needed for the different parts of the system
    generator = FileGenerator()
    # generator.generate_hotel_list() # stored in path=../generated_files/hotel_list/hotels_list.csv
    # generator.generate_pos_neg_reviews_of_hotel() # stored in path=../generated_files/hotel_pos_neg_reviews/<hotel_name>.csv
    # generator.generate_noun_adj_dict() # stored in path=../generated_files/hotel_<pos|neg>_review_noun_adj_dict/<hotel_name>.pickle
    # generator.generate_best_features_term_frequency_approach() # stored in path=../generated_files/hotel_best_features_term_freq/<hotel_name>.pickle
    # generator.generate_worst_features_term_frequency_approach() # stored in path=../generated_files/hotel_worst_features_term_freq/<hotel_name>.pickle
    # generator.generate_training_set_for_sentiment_clf_decision_tree() # stored in path=../generated_files/sentiment_clf_DecisionTreeClassifier/X_y_training_set.pickle
    # generator.generate_fit_training_set_using_tfidf_vectorizer() # stored in path=../generated_files/sentiment_clf_DecisionTreeClassifier/X_tfidf_y_training_set.pickle
    # generator.generate_sentiment_clf_decision_tree() # stored in path=../generated_files/sentiment_clf_DecisionTreeClassifier/Decision_Tree_classifier.pickle
    # generator.generate_average_scores() # stored in path=../generated_files/hotel_report/
    generator.generate_hotel_address() # stored in path=../generated_files/hotel_report/
generate_files()
# feature_extraction_method_2()

# simple_recommend()

# "hioshgo"  0
# "kaerngd"  1
# ["dsg"]    0
# ["dsaf", "sadg"]  1


########
# Main #
########
# Functionality of this script:
# 1) Build a bag of words (manually) containing the features 
# that people would generally comment on in a hotel review.
# 2) For each feature, select all reviews that mentions it
# 3) Train classifiers to identify each of the feature (returned)
# 4) Find the most common words that describes these features, or what's good about them
# which is going to be used in the knowledge graph (returned)
# =======================================================================================

# UI for seeing past reviews on features
# start()
# recommend()

# training classifiers for recognising good/bad reviews of each feature [0] = positive, [1] = negative
# trian_feature_classifiers()

# read_in_feature_clf("room")
# train_pos_neg_review_clf()

# cat = Categorical()
# cat.labelling()
# cat.hotel_with_most_labels()
# cat.labelling_negatives()



