from django.db import models

import pandas as pd
import pickle

# Create your models here.
def getHotels():
    unique_hotel_list = pd.read_csv("../../generated_files/hotel_list/hotels_list.csv", usecols=['Hotel_Name'])
    unique_hotel_list = unique_hotel_list.Hotel_Name.unique()
    return unique_hotel_list

def get_average_ratings(hotel_selected):
    pickle_in = open("../../generated_files/hotel_report/hotel_average_score.pickle", "rb")
    df = pickle.load(pickle_in)
    name_rating = df.loc[df['Hotel_Name']==hotel_selected]
    return name_rating['Average_score'].values

def get_hotel_address(hotel_selected):
    pickle_in = open("../../generated_files/hotel_report/hotel_address.pickle", "rb")
    df = pickle.load(pickle_in)
    name_rating = df.loc[df['Hotel_Name']==hotel_selected]
    return name_rating['Hotel_Address'].values

def get_feature_comments(hotel_selected):
    pickle_in = open("../../generated_files/hotel_best_features_term_freq/{}.pickle".format(hotel_selected), "rb")
    best_features = pickle.load(pickle_in)

    feature_info = []
    for k,v in best_features.items():
        info = []
        info.append(k)
        info.append(list(v.items()))
        feature_info.append(info)
    
    pickle_in = open("../../generated_files/hotel_worst_features_term_freq/{}.pickle".format(hotel_selected), "rb")
    worst_features_list = pickle.load(pickle_in)

    worst_features = []
    for k,v in worst_features_list.items():
        info = []
        info.append(k)
        info.append(list(v.items()))
        worst_features.append(info)
    
    return feature_info, worst_features

def get_nationality_preferences(hotel_selected):
    full_dataset = pd.read_csv("../../data/Hotel_Reviews.csv")
    # favoured by customers from
    pos_name_nationality_score = full_dataset[['Hotel_Name', 'Reviewer_Nationality', 'Reviewer_Score']].copy()
    pos_name_nationality_score = pos_name_nationality_score.loc[pos_name_nationality_score['Hotel_Name']==hotel_selected]
    pos_nationality_score = pos_name_nationality_score.groupby('Reviewer_Nationality', as_index=False).mean().sort_values(by='Reviewer_Score', ascending=False)
    pos_nationality_score = pos_nationality_score.loc[pos_nationality_score['Reviewer_Score'] >= 8,:]
    pos_nationality_score = pos_nationality_score.values.tolist()
    
    # disliked by customers from
    neg_name_nationality_score = full_dataset[['Hotel_Name', 'Reviewer_Nationality', 'Reviewer_Score']].copy()
    neg_name_nationality_score = neg_name_nationality_score.loc[neg_name_nationality_score['Hotel_Name']==hotel_selected]
    neg_nationality_score = neg_name_nationality_score.groupby('Reviewer_Nationality',as_index=False).mean().sort_values(by='Reviewer_Score', ascending=False)
    neg_nationality_score = neg_nationality_score.loc[neg_nationality_score['Reviewer_Score'] <= 3,:]
    neg_nationality_score = neg_nationality_score.values.tolist()

    return pos_nationality_score, neg_nationality_score