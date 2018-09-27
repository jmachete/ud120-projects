from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### create classifier
    classifier = GaussianNB()

    ### fit the classifier on the training features and labels
    classifier.fit(features_train, labels_train)

    ### use the trained classifier to predict labels for the test features
    pred = classifier.predict(features_test)

    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example,
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = accuracy_score(pred, labels_test)

    return accuracy
