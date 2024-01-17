#%%
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def naive_bayes_function(X_train, X_test, y_train, y_test):
    # Initialize Gaussian Naive Bayes classifier
    nb_clf = GaussianNB()

    # Fit the model on the training data
    nb_clf.fit(X_train, y_train)

    # Predict probabilities for train and test sets
    train_probs = nb_clf.predict_proba(X_train)
    test_probs = nb_clf.predict_proba(X_test)

    # Calculate log loss for train and test sets
    train_log_loss = log_loss(y_train, train_probs, labels=nb_clf.classes_, eps=1e-15)
    test_log_loss = log_loss(y_test, test_probs, labels=nb_clf.classes_, eps=1e-15)

    # Display log loss for train and test sets
    print(f'Train Log Loss: {train_log_loss:.5f}')
    print(f'Test Log Loss: {test_log_loss:.5f}')

    # Predict labels for the test set
    predicted_labels = nb_clf.predict(X_test)

    # Display confusion matrix
    cm = confusion_matrix(y_test, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb_clf.classes_)
    disp.plot(cmap='Blues', values_format='d')

    plt.title('Confusion Matrix')
    plt.show()

# Example usage:
# naive_bayes_function(X_train, X_test, y_train, y_test)
