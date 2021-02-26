import datetime
import pickle
from tqdm import tqdm
from Analyser import Trainer
from DataProcessor import FeatureReviewsExtractor, DataPreprocessing
from DataProvider import Getter


class CLFtrainer:

    def train_pos_neg_review_clf(self):
        pickle_in = open("../pickled_files/reviews.pickle","rb")
        df = pickle.load(pickle_in)

        pos_col = df["Positive_Review"]
        neg_col = df["Negative_Review"]

        # Clean reviews
        print("##########################################")
        now = datetime.datetime.now()
        print("{} : Start cleaning".format(now.strftime("%Y-%m-%d %H:%M:%S")))
        processor = DataPreprocessing()

        cleaned_pos_reviews = processor.clean_without_feature(pos_col)
        now = datetime.datetime.now()
        print("{} : finished cleaning positive reviews".format(now.strftime("%Y-%m-%d %H:%M:%S")))

        cleaned_neg_reviews = processor.clean_without_feature(neg_col)
        now = datetime.datetime.now()
        print("{} : finished cleaning negative reviews".format(now.strftime("%Y-%m-%d %H:%M:%S")))

        now = datetime.datetime.now()
        print("{} : finished cleaning".format(now.strftime("%Y-%m-%d %H:%M:%S")))
        print("##########################################")

        # Construct data set for training
        X, y = processor.construct_feature_set(cleaned_pos_reviews, cleaned_neg_reviews)

        trainer = Trainer()
        vectorizer, clf = trainer.pos_neg_clf(X, y)

        return vectorizer, clf

    def trian_feature_classifiers(self): 
        pickle_in = open("../pickled_files/reviews.pickle","rb")
        df = pickle.load(pickle_in)

        getter = Getter()
        features = getter.getFeatureList()

        # Extract reviews that mentions the feature
        extractor = FeatureReviewsExtractor()
        for i in tqdm(range(0,len(features))):
            positive_review, negative_review = extractor.positive_negative_reviews_on_feature(df, features[i])
            print(features[i])
            # Clean reviews
            processor = DataPreprocessing()
            cleaned_pos_reviews, cleaned_neg_reviews = processor.preprocessing_3(positive_review, negative_review, features[i])

            # Construct date set for training
            X, y = processor.construct_feature_set(cleaned_pos_reviews, cleaned_neg_reviews)

            # Train classifier
            analyser = Trainer()
            parameters = ["feature_clfs_tree",features[i]]
            clf = analyser.train(X, y, parameters)
            pickle_out = open("../pickled_clfs/{}.pickle".format(features[i]),"wb")
            pickle.dump(clf, pickle_out)
            pickle_out.close()
            break