import pandas as pd
import os
import pickle
from tqdm import tqdm
import re
import glob
from DataProcessor import FeatureReviewsExtractor, DataPreprocessing
from Analyser import Trainer
from DataProvider import Getter
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









# path = "../data/Hotel_Reviews.csv"
# columns_to_use = ["Hotel_Name","Positive_Review", "Negative_Review"]
# df = read_file(path, columns_to_use)
# pickle_out = open("../pickled_files/reviews.pickle","wb")
# pickle.dump(df, pickle_out)
# pickle_out.close()
# pickle_in = open("../pickled_files/reviews.pickle","rb")
# df = pickle.load(pickle_in)

# hotel_list = df.Hotel_Name.unique()
# sub_list = hotel_list[:3] # experiment with 5 hotels first

# features = bag_of_words()

# for each hotel, extract the different adj terms used in pos and neg reviews of a feature
# hotel = {}
# for i in tqdm(range(0,len(sub_list))):
#     hotel_info = df.loc[df['Hotel_Name'] == sub_list[i]]
    
#     terms = {}
#     for n in range(0,len(features[:3])):
#         pos_terms, neg_terms = feature_extraction(hotel_info, features[n])
#         terms[features[n]] = {'positive': pos_terms,
#                               'negative': neg_terms}

#     hotel[sub_list[i]] = {"features": terms}



