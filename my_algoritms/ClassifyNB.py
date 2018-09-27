from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):
    
    ### create classifier
    classifier = GaussianNB()

    ### fit the classifier on the training features and labels
    classifier.fit(features_train, labels_train)

    return classifier
