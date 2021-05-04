import pickle
import pandas as pd
import re
import nltk
import numpy as np
import datetime
import matplotlib.pyplot as plt
# import wandb

from os import path
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from DataProvider import Getter
from Analyser import Trainer

def text_cleaning(X):
    stemmer = WordNetLemmatizer()
    sentences = []

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

def prepare_data():
    pickle_in = open("../pickled_files/reviews.pickle","rb")
    df = pickle.load(pickle_in)
    pos_col = df["Positive_Review"]
    neg_col = df["Negative_Review"]

    cleaned_positive_reviews = text_cleaning(pos_col)
    cleaned_negative_reviews = text_cleaning(neg_col)

    pickle_out = open("../pickled_files/cleaned_positive_reviews.pickle","wb")
    pickle.dump(cleaned_positive_reviews, pickle_out)
    pickle_out.close()

    pickle_out = open("../pickled_files/cleaned_negative_reviews.pickle","wb")
    pickle.dump(cleaned_negative_reviews, pickle_out)
    pickle_out.close()

def check_data_availablility():
    cleaned_pos = "../pickled_files/cleaned_positive_reviews.pickle"
    cleaned_neg = "../pickled_files/cleaned_negative_reviews.pickle"

    if (not path.exists(cleaned_pos) or not path.exists(cleaned_neg)):
        prepare_data()

def construct_data_set(X_target, y_other):
    y_target_label = [0] * len(X_target)
    y_other_label = [1] * len(y_other)

    X = X_target+y_other
    y = y_target_label+y_other_label
    print("X_target size = {}".format(len(X_target)))
    print("y_other size = {}".format(len(y_other)))
    print("X size = {}".format(len(X)))
    print("y size = {}".format(len(y)))

    pickle_out = open("../pickled_files/X_training_set.pickle","wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("../pickled_files/y_training_set_label.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    return X, y

def check_if_training_set_constructed():
    X_path = "../pickled_files/X_training_set.pickle"
    y_path = "../pickled_files/y_training_set_label.pickle"

    if (not path.exists(X_path) or not path.exists(y_path)):
        getter = Getter()
        cleaned_positive_reviews = getter.get_cleaned_reviews("positive")
        cleaned_negative_reviews = getter.get_cleaned_reviews("negative")
        X, y = construct_data_set(cleaned_positive_reviews, cleaned_negative_reviews)
        return X, y
    else:
        pickle_in = open(X_path, "rb")
        X = pickle.load(pickle_in)
        pickle_in = open(y_path, "rb")
        y = pickle.load(pickle_in)
        return X, y

def check_if_bag_of_words_package_exist(X, y):
    package_path = "../pickled_clfs/bag_of_words_package.pickle"

    if (not path.exists(package_path)):
        # create bag of words / document term matrix
        print("building new bag of words package...")
        trainer = Trainer()
        max_features = 20
        min_df = 0.25
        info = "bag_of_words_package"
        X_train, X_test, y_train, y_test, vectorizer, X_transformed = trainer.create_vectorizer(X, y, max_features, min_df, info)
        return X_train, X_test, y_train, y_test, vectorizer, X_transformed
    else:
        print("bag of words package exist...")
        pickle_in = open(package_path, "rb")
        package = pickle.load(pickle_in)
        X_train = package[0]
        X_test = package[1]
        y_train = package[2]
        y_test = package[3]
        vectorizer = package[4]
        X_transformed = package[5]
        return X_train, X_test, y_train, y_test, vectorizer, X_transformed

def create_document_term_matrix(X_transformed, vectorizer):
    return pd.DataFrame(X_transformed, columns=vectorizer.get_feature_names())

def model_selection(clf, X_train, X_test, y_train, y_test):
    max_depths = np.linspace(1, 10, 5, endpoint=True)
    param_dist={
        "criterion":["gini", "entropy"],
        "max_depth":max_depths
    }
    
    grid = GridSearchCV(clf, param_grid=param_dist, cv=5, n_jobs=-1, verbose=1, refit=True)
    grid.fit(X_train, y_train)

    pickle_out = open("../pickled_files/grid.pickle", "wb")
    pickle.dump(grid, pickle_out)
    pickle_out.close()
    print("grid object saved...")
    print("--------------------------------")
    
    print("best estimator = {}".format(grid.best_estimator_))
    print("best parameters = {}".format(grid.best_params_))

    y_pred = grid.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))
    print("best score = {}".format(grid.best_score_))

def evaluate_bag_of_words_max_features():
    print("------------------------------------------")
    start = datetime.datetime.now()
    print("{} : Begin process...".format(start.strftime("%Y-%m-%d %H:%M:%S")))

    trainer = Trainer()
    
    X_path = "../pickled_files/X_training_set.pickle"
    y_path = "../pickled_files/y_training_set_label.pickle"
    pickle_in = open(X_path, "rb")
    X = pickle.load(pickle_in)
    pickle_in = open(y_path, "rb")
    y = pickle.load(pickle_in)

    train_accuracy = []
    test_accuracy = []
    max_features = [20, 50, 100, 200]
    for max_feature in max_features:
        info = "{}_feature_vector".format(str(max_feature))
        min_df = None
        
        X_train, X_test, y_train, y_test, vectorizer, X_transformed = trainer.create_vectorizer(X, y, max_feature, min_df, info)

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(criterion="gini", max_depth=10)
        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        # predict training dataset
        y_train_pred =clf.predict(X_train)

        # Predict the response for test dataset
        y_test_pred = clf.predict(X_test)

        train_accuracy.append(metrics.accuracy_score(y_train, y_train_pred))
        test_accuracy.append(metrics.accuracy_score(y_test, y_test_pred))

    print("show plot...")
    plt.plot(max_features, train_accuracy, label="Train")
    plt.plot(max_features, test_accuracy, label="Test")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of features")
    plt.legend()
    plt.title("Accuracy of predictions with different number of feature selections")
    plt.show()

    print("------------------------------------------")
    end = datetime.datetime.now()
    print("{} : Process completed...".format(end.strftime("%Y-%m-%d %H:%M:%S")))
    print("Time taken : {}".format(end - start))

def decision_tree_model_selection_by_gridCV():
    print("------------------------------------------")
    start = datetime.datetime.now()
    print("{} : Begin process...".format(start.strftime("%Y-%m-%d %H:%M:%S")))


    check_data_availablility()
    print("data is prepared...")

    # load X and y training set
    X, y = check_if_training_set_constructed()
    print("training set is constructed and saved...")

    # load bag of words package :: vectorizer, train/test sets
    X_train, X_test, y_train, y_test, vectorizer, X_transformed = check_if_bag_of_words_package_exist(X, y)
    print("built vectorizer...")
    print("number of features used = {}".format(len(vectorizer.get_feature_names())))
    print(vectorizer.vocabulary_)

    document_term_matrix = create_document_term_matrix(X_transformed, vectorizer)
    print(document_term_matrix.head(10))

    # training decision tree classifier on default settings
    trainer = Trainer()
    trainer.sentiment_clf(X_train, X_test, y_train, y_test, vectorizer.get_feature_names())

    # using GridSearchCV to find the best parameters of the classifier model
    pickle_in = open("../pickled_clfs/default_pos_neg_review_clf.pickle", "rb")
    clf = pickle.load(pickle_in)
    model_selection(clf, X_train, X_test, y_train, y_test)

    print("------------------------------------------")
    end = datetime.datetime.now()
    print("{} : Process completed...".format(end.strftime("%Y-%m-%d %H:%M:%S")))
    print("Time taken : {}".format(end - start))

def best_model():
    print("------------------------------------------")
    start = datetime.datetime.now()
    print("{} : Begin process...".format(start.strftime("%Y-%m-%d %H:%M:%S")))

    trainer = Trainer()
    
    X_path = "../pickled_files/X_training_set.pickle"
    y_path = "../pickled_files/y_training_set_label.pickle"
    pickle_in = open(X_path, "rb")
    X = pickle.load(pickle_in)
    pickle_in = open(y_path, "rb")
    y = pickle.load(pickle_in)

    max_feature = 50
    info = "50_feature_vector"
    min_df = None
    X_train, X_test, y_train, y_test, vectorizer, X_transformed = trainer.create_vectorizer(X, y, max_feature, min_df, info)    

    pickle_out = open("../best_clfs_tree/best_pos_neg_review_vectorizer.pickle","wb")
    pickle.dump(vectorizer, pickle_out)
    pickle_out.close()

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="gini", max_depth=10)
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    pickle_out = open("../best_clfs_tree/best_pos_neg_review_clf.pickle","wb")
    pickle.dump(clf, pickle_out)
    pickle_out.close()

    print("Clf stored successfully!")
    print("------------------------------------------")

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print(metrics.classification_report(y_test,y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    export_graphviz(clf, out_file="../{}/{}.dot".format("best_clfs_tree", "pos_neg_review"),
                    filled=True, rounded=True,
                    special_characters=True,
                    feature_names = vectorizer.get_feature_names(),
                    class_names=['positive','nagative'])
    print("------------------------------------------")

def SVM():
    print("------------------------------------------")
    start = datetime.datetime.now()
    print("{} : Begin process...".format(start.strftime("%Y-%m-%d %H:%M:%S")))

    trainer = Trainer()
    
    X_path = "../pickled_files/X_training_set.pickle"
    y_path = "../pickled_files/y_training_set_label.pickle"
    pickle_in = open(X_path, "rb")
    X = pickle.load(pickle_in)
    pickle_in = open(y_path, "rb")
    y = pickle.load(pickle_in)

    data_range = [5000, 10000, 15000, 20000, 25000]
    x_axis_points = [x*2 for x in data_range]
    
    train_accuracy = []
    test_accuracy = []

    for n in data_range:

        small_X = X[:n] + X[-n:]
        small_y = y[:n] + y[-n:]

        max_feature = 50
        info = "50_feature_vector_SVM"
        min_df = None
        X_train, X_test, y_train, y_test, vectorizer, X_transformed = trainer.create_vectorizer(small_X, small_y, max_feature, min_df, info)
        print("---------Test/Train set ready-------------------")

        clf = svm.SVC(kernel='linear', random_state=1)
        clf.fit(X_train, y_train)

        print("SVM Clf built")
        print("------------------------------------------")

        pickle_out = open("../SVM/pos_neg_SVM_clf_{}.pickle".format(n), "wb")
        pickle.dump(clf, pickle_out)
        pickle_out.close
        print("SVM Clf saved")
        print("------------------------------------------")

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        print(metrics.classification_report(y_test,y_pred))
        test_acc = metrics.accuracy_score(y_test, y_pred)
        print("Accuracy:",test_acc)
        test_accuracy.append(test_acc)

        # predict the resoinse for training dataset
        y_train_pred = clf.predict(X_train)
        train_acc = metrics.accuracy_score(y_train, y_train_pred)
        train_accuracy.append(train_acc)

        print("------------------------------------------")
        end = datetime.datetime.now()
        print("{} : Process completed...".format(end.strftime("%Y-%m-%d %H:%M:%S")))
        print("Time taken : {}".format(end - start))
        print("------------------------------------------")
    
    print("show plot...")
    plt.plot(x_axis_points, train_accuracy, label="Train")
    plt.plot(x_axis_points, test_accuracy, label="Test")
    plt.ylabel("Accuracy")
    plt.xlabel("Samples used")
    plt.legend()
    plt.title("Accuracy of SVM predictions with different number of samples used")
    plt.show()

def multiple_classifiers_comparison():
    print("------------------------------------------")
    start = datetime.datetime.now()
    print("{} : Begin process...".format(start.strftime("%Y-%m-%d %H:%M:%S")))

    trainer = Trainer()
    
    X_path = "../pickled_files/X_training_set.pickle"
    y_path = "../pickled_files/y_training_set_label.pickle"
    pickle_in = open(X_path, "rb")
    X = pickle.load(pickle_in)
    pickle_in = open(y_path, "rb")
    y = pickle.load(pickle_in)

    data_range = [5000, 10000, 15000, 20000, 25000]
    x_axis_points = [x*2 for x in data_range]
    
    names = ["Decision Tree", "SVM", "Logistic Regression", "Random Forest", " Gaussian Naive Bayes"]

    decision_tree = DecisionTreeClassifier(criterion="gini", random_state=1)
    SVM = svm.SVC(kernel='linear', random_state=1)
    logistic_regression = LogisticRegression(random_state=1)
    random_forest = RandomForestClassifier(random_state=1)
    gnb = GaussianNB()

    classifiers = [decision_tree, SVM, logistic_regression, random_forest, gnb]

    accuracy = []

    for clf in classifiers:

        acc_list = []
        for n in data_range:
            small_X = X[:n] + X[-n:]
            small_y = y[:n] + y[-n:]

            max_feature = 50
            info = "50_feature_vector"
            min_df = None
            X_train, X_test, y_train, y_test, vectorizer, X_transformed = trainer.create_vectorizer(small_X, small_y, max_feature, min_df, info)
            print("---------Test/Train set ready-------------------")

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = metrics.accuracy_score(y_test, y_pred)
            acc_list.append(acc)
        print("----------Next Classifier---------")

        accuracy.append(acc_list)
    
    print("------------------------------------------")
    end = datetime.datetime.now()
    print("{} : Process completed...".format(end.strftime("%Y-%m-%d %H:%M:%S")))
    print("Time taken : {}".format(end - start))
    print("------------------------------------------")

    print("show plot...")
    for index, val in enumerate(accuracy):
        plt.plot(x_axis_points, val, label=names[index])
        
    plt.ylabel("Accuracy")
    plt.xlabel("Samples used")
    plt.legend()
    plt.title("Accuracy of predictions with 5 different classifiers")
    plt.show()




stop_words = nltk.corpus.stopwords.words('english')
stop_words = set(stopwords.words('english')) - set(['no'])

# main()
# evaluate_bag_of_words_max_features()
# best_model()
# SVM()
# multiple_classifiers_comparison()

