import pandas as pd
import pickle

class Getter:

    def getHotelList(self):
        df = pd.read_csv("../data/Hotel_Reviews.csv", usecols=['Hotel_Name'])
        unique_hotel_list = df.Hotel_Name.unique()
        return unique_hotel_list

    def getFeatureList(self):
        features = ["room","bed","location","bathroom","staff","staircase",
                    "park","hotel","building","style","transport","parking",
                    "food","breakfast","lunch","dinner","restaurant"]
        return features

    def get_cleaned_reviews(self, review_type):
        path = "../pickled_files/cleaned_{}_reviews.pickle".format(review_type)
        pickle_in = open(path,"rb")
        cleaned_reviews = pickle.load(pickle_in)
        return cleaned_reviews

    def getPosNegReviewsClf(self):
        pickle_in = open("../pickled_clfs/pos_neg_review_vectorizer.pickle","rb")
        vectorizer = pickle.load(pickle_in)
        pickle_in = open("../pickled_clfs/pos_neg_review_clf.pickle","rb")
        clf = pickle.load(pickle_in)
        return vectorizer, clf
    
    def getCategories(self):
        categories = ["clean_room","nice_breakfast","good_location","friendly_staff","comfortable_bed",
                      "value_for_money","parking", "wifi","nice_gym"]
        return categories
    
    def getFullDataset(self):
        hotel_list = pd.read_csv("../data/Hotel_Reviews.csv")
        return hotel_list
    
    def getSelectedCols(self, attributes):
        data = pd.read_csv("../data/Hotel_Reviews.csv", usecols=attributes)
        return data