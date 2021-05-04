import pandas as pd
import pickle
from tqdm import tqdm


class Getter:

    def getHotelList(self):
        unique_hotel_list = pd.read_csv("../generated_files/hotel_list/hotels_list.csv", usecols=['Hotel_Name'])
        return unique_hotel_list

    def get_pos_neg_reviews_of_hotel(self, hotel_name):
        df = pd.read_csv("../generated_files/hotel_pos_neg_reviews/{}.csv".format(hotel_name))
        return df

    def get_pos_noun_adj_dict_of_hotel(self, hotel_name):
        pickle_in = open("../generated_files/hotel_pos_review_noun_adj_dict/{}.pickle".format(hotel_name), "rb")
        pos_dict = pickle.load(pickle_in)
        return pos_dict
    
    def get_neg_noun_adj_dict_of_hotel(self, hotel_name):
        pickle_in = open("../generated_files/hotel_neg_review_noun_adj_dict/{}.pickle".format(hotel_name), "rb")
        neg_dict = pickle.load(pickle_in)
        return neg_dict

    def get_best_features_term_frequency_approach(self, hotel_name):
        pickle_in = open("../generated_files/hotel_best_features_term_freq/{}.pickle".format(hotel_name), "rb")
        best_features = pickle.load(pickle_in)
        return best_features
    
    def get_worst_features_term_frequency_approach(self, hotel_name):
        pickle_in = open("../generated_files/hotel_worst_features_term_freq/{}.pickle".format(hotel_name), "rb")
        worst_features = pickle.load(pickle_in)
        return worst_features
    
    def get_training_set_for_sentiment_clf_decision_tree(self):
        pickle_in = open("../generated_files/sentiment_clf_DecisionTreeClassifier/X.pickle", "rb")
        X = pickle.load(pickle_in)

        pickle_in = open("../generated_files/sentiment_clf_DecisionTreeClassifier/y.pickle", "rb")
        y = pickle.load(pickle_in)
        return X, y
    
    def get_X_tfidf_y_training_set_for_sentiment_clf_decision_tree(self):
        pickle_in = open("../generated_files/sentiment_clf_DecisionTreeClassifier/X_tfidf.pickle", "rb")
        X_tfidf = pickle.load(pickle_in)

        pickle_in = open("../generated_files/sentiment_clf_DecisionTreeClassifier/y.pickle", "rb")
        y = pickle.load(pickle_in)

        pickle_in = open("../generated_files/sentiment_clf_DecisionTreeClassifier/tfIdfVectorizer.pickle", "rb")
        tfIdfVectorizer = pickle.load(pickle_in)
        return X_tfidf, y, tfIdfVectorizer
    
    def get_sentiment_decision_tree_clf(self):
        pickle_in = open("../generated_files/sentiment_clf_DecisionTreeClassifier/Decision_Tree_classifier.pickle", "rb")
        clf = pickle.load(pickle_in)

        pickle_in = open("../generated_files/sentiment_clf_DecisionTreeClassifier/tfIdfVectorizer.pickle", "rb")
        tfIdfVectorizer = pickle.load(pickle_in)
        return clf, tfIdfVectorizer

    def getFeatureList(self):
        features = ["room","bed","location","bathroom","staff","staircase",
                    "park","hotel","building","style","transport","parking",
                    "food","breakfast","lunch","dinner","restaurant", "bar"]
        return features

    def get_cleaned_reviews(self, review_type):
        path = "../pickled_files/cleaned_{}_reviews.pickle".format(review_type)
        pickle_in = open(path,"rb")
        cleaned_reviews = pickle.load(pickle_in)
        return cleaned_reviews

    def getPosNegReviewsClf(self):
        pickle_in = open("../best_clfs_tree/best_pos_neg_review_vectorizer.pickle","rb")
        vectorizer = pickle.load(pickle_in)
        pickle_in = open("../best_clfs_tree/best_pos_neg_review_clf.pickle","rb")
        clf = pickle.load(pickle_in)
        return vectorizer, clf
    
    def getCategories(self):
        categories = ["clean_room","nice_breakfast","good_location","friendly_staff","comfortable_bed",
                      "value_for_money","parking", "wifi","nice_gym", "good_transport", "nice_bar"]
        return categories
    
    def getFullDataset(self):
        hotel_list = pd.read_csv("../data/Hotel_Reviews.csv")
        return hotel_list
    
    def getSelectedCols(self, attributes):
        data = pd.read_csv("../data/Hotel_Reviews.csv", usecols=attributes)
        return data


class FileGenerator:
    def generate_hotel_list(self):
        print("generate_hotel_list...",end='')
        df = pd.read_csv("../data/Hotel_Reviews.csv", usecols=['Hotel_Name'])
        unique_hotel_list = df.Hotel_Name.unique()
        unique_hotel_list = pd.DataFrame(unique_hotel_list, columns=['Hotel_Name'])
        unique_hotel_list.to_csv("../generated_files/hotel_list/hotels_list.csv")
        print("Done")

    def generate_pos_neg_reviews_of_hotel(self):
        print("generate_pos_neg_reviews_of_hotel...",end='')
        getter = Getter()
        unique_hotel_list = getter.getHotelList()
        df = pd.read_csv("../data/Hotel_Reviews.csv", usecols=['Hotel_Name','Negative_Review','Positive_Review'])

        for hotel in tqdm(unique_hotel_list.values):
            hotel_name = hotel[0]
            hotel_df = df.loc[df['Hotel_Name']==hotel_name]
            hotel_df.to_csv("../generated_files/hotel_pos_neg_reviews/{}.csv".format(hotel_name))
        print("Done")

    def generate_noun_adj_dict(self):
        print("generate_noun_adj_dictionaires...", end='')
        from DataProcessor import TextAnalysis
        getter = Getter()
        feature_extractor = TextAnalysis()
        hotel_list = getter.getHotelList()

        # Loop through each hotel, perform text analysis on its reviews, store in dictionary format
        for hotel in tqdm(hotel_list.values):
            hotel_name = hotel[0]
            df = getter.get_pos_neg_reviews_of_hotel(hotel_name)
            
            # Positive reviews
            positive_reviews = df["Positive_Review"].values.tolist()
            positive_dict = self.generate_dictionary(positive_reviews, feature_extractor, "pos")
            # store
            pickle_out = open("../generated_files/hotel_pos_review_noun_adj_dict/{}.pickle".format(hotel_name),"wb")
            pickle.dump(positive_dict, pickle_out)
            pickle_out.close()
            # read
            # pickle_in = open("../generated_files/hotel_pos_review_noun_adj_dict/{}.pickle".format(hotel_name), "rb")
            # pos = pickle.load(pickle_in)
            # print("POSITIVE == ",end='')
            # print(pos)


            # Negative reviews
            negative_reviews = df["Negative_Review"].values.tolist()
            negative_dict = self.generate_dictionary(negative_reviews, feature_extractor, "neg")
            # store
            pickle_out = open("../generated_files/hotel_neg_review_noun_adj_dict/{}.pickle".format(hotel_name),"wb")
            pickle.dump(negative_dict, pickle_out)
            pickle_out.close()
            # read
            # pickle_in = open("../generated_files/hotel_neg_review_noun_adj_dict/{}.pickle".format(hotel_name), "rb")
            # neg = pickle.load(pickle_in)
            # print("NEGATIVE == ",end='')
            # print(neg)
            # break
        print("Done")
    
    def generate_dictionary(self, reviews, feature_extractor, review_type):
        dict_ = {}
        
        for sentence in reviews:
            dictionary = feature_extractor.start(sentence,review_type)
            
            # ignore empty result
            if not dictionary:
                continue

            for k, v in dictionary.items():
                if k in dict_:
                    dict_[k] = dict_[k] + v
                else:
                    dict_[k] = v
            # break
        return dict_
    
    def generate_best_features_term_frequency_approach(self):
        # Extract the best 5 features only, with their corresponding values
        from Analyser import FeatureExtractor
        print("generate_best_features_term_frequency_approach...", end='')
        feature_extractor = FeatureExtractor()
        getter = Getter()
        hotel_list = getter.getHotelList()

        for hotel in tqdm(hotel_list.values):
            hotel_name = hotel[0]
            pos_dict = getter.get_pos_noun_adj_dict_of_hotel(hotel_name)

            best_features = feature_extractor.best_features_term_frequency_approach(hotel_name, pos_dict)

            pickle_out = open("../generated_files/hotel_best_features_term_freq/{}.pickle".format(hotel_name),"wb")
            pickle.dump(best_features, pickle_out)
            pickle_out.close()

            # print(str(hotel_name)+" = ",end="")
            # pickle_in = open("../generated_files/hotel_best_features_term_freq/{}.pickle".format(hotel_name), "rb")
            # best_features = pickle.load(pickle_in)
            # print(best_features)
        print("Done")
    
    def generate_worst_features_term_frequency_approach(self):
        from Analyser import FeatureExtractor
        print("generate_worst_features_term_frequency_approach...", end='')
        feature_extractor = FeatureExtractor()
        getter = Getter()
        hotel_list = getter.getHotelList()

        for hotel in tqdm(hotel_list.values):
            hotel_name = hotel[0]
            neg_dict = getter.get_neg_noun_adj_dict_of_hotel(hotel_name)

            worst_features = feature_extractor.best_features_term_frequency_approach(hotel_name, neg_dict)

            pickle_out = open("../generated_files/hotel_worst_features_term_freq/{}.pickle".format(hotel_name),"wb")
            pickle.dump(worst_features, pickle_out)
            pickle_out.close()

            # print(str(hotel_name)+" = ",end="")
            # pickle_in = open("../generated_files/hotel_worst_features_term_freq/{}.pickle".format(hotel_name), "rb")
            # worst_features = pickle.load(pickle_in)
            # print(best_features)
        print("Done")
    
    def generate_training_set_for_sentiment_clf_decision_tree(self):
        from Analyser import Trainer
        print("generate_training_set_for_sentiment_clf_decision_tree...", end='')
        trainer = Trainer()

        trainer.form_training_set_for_sentiment_clf_decision_tree()
        X, y = trainer.form_training_set_for_sentiment_clf_decision_tree()

        # X = 524115 samples in total, 285723 positive [0] samples, and 238392 negative [1] samples

        pickle_out = open("../generated_files/sentiment_clf_DecisionTreeClassifier/X.pickle","wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("../generated_files/sentiment_clf_DecisionTreeClassifier/y.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
        
        # pickle_in = open("../generated_files/sentiment_clf_DecisionTreeClassifier/X_y_training_set.pickle", "rb")
        # training = pickle.load(pickle_in)
        print("Done")

    def generate_fit_training_set_using_tfidf_vectorizer(self):
        from Analyser import Trainer
        print("generate_fit_training_set_using_tfidf_vectorizer...")
        trainer = Trainer()
        X_tfidf, tfIdfVectorizer = trainer.fit_training_set_using_tfidf_vectorizer()

        pickle_out = open("../generated_files/sentiment_clf_DecisionTreeClassifier/X_tfidf.pickle","wb")
        pickle.dump(X_tfidf, pickle_out)
        pickle_out.close()

        pickle_out = open("../generated_files/sentiment_clf_DecisionTreeClassifier/tfIdfVectorizer.pickle","wb")
        pickle.dump(tfIdfVectorizer, pickle_out)
        pickle_out.close()
        print("Done")

    def generate_sentiment_clf_decision_tree(self):
        from Analyser import Trainer
        print("generate_sentiment_decision_tree_clf...")
        trainer = Trainer()
        clf = trainer.train_sentiment_clf_decision_tree()

        pickle_out = open("../generated_files/sentiment_clf_DecisionTreeClassifier/Decision_Tree_classifier.pickle","wb")
        pickle.dump(clf, pickle_out)
        pickle_out.close()

        print("Done")