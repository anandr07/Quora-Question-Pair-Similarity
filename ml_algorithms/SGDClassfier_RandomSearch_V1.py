from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import log_loss, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def sgd_random_search_v1(X_train, X_test, y_train, y_test):
    # Hyperparameter grid for RandomizedSearchCV
    param_dist = {
        'base_estimator__alpha': [10 ** x for x in range(-5, 2)],
        'base_estimator__penalty': ['l1', 'l2', 'elasticnet'],
        'base_estimator__loss': ['log', 'modified_huber'],
        'method': ['sigmoid']
    }

    # Initialize SGDClassifier with penalty
    sgd_clf = SGDClassifier(random_state=42, penalty='elasticnet')

    # Initialize CalibratedClassifierCV
    calibrated_clf = CalibratedClassifierCV(base_estimator=sgd_clf)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        calibrated_clf, 
        param_distributions=param_dist, 
        scoring='neg_log_loss', 
        n_iter=10,  # Adjust the number of iterations as needed
        cv=3, 
        n_jobs=-1
    )

    # Perform RandomizedSearchCV on the data
    random_search.fit(X_train, y_train)

    # Print the best parameters
    print("Best Parameters:", random_search.best_params_)

    # Get the best model from RandomizedSearchCV
    best_model = random_search.best_estimator_

    # Predict probabilities for train and test sets
    train_probs = best_model.predict_proba(X_train)
    test_probs = best_model.predict_proba(X_test)

    # Calculate log loss for train and test sets
    train_log_loss = log_loss(y_train, train_probs, labels=best_model.classes_, eps=1e-15)
    test_log_loss = log_loss(y_test, test_probs, labels=best_model.classes_, eps=1e-15)

    # Display log loss for train and test sets
    print(f'Train Log Loss: {train_log_loss:.5f}')
    print(f'Test Log Loss: {test_log_loss:.5f}')

    # Predict labels for the test set
    predicted_labels = best_model.predict(X_test)

    # Display confusion matrix
    cm = confusion_matrix(y_test, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap='Blues', values_format='d')

    plt.title('Confusion Matrix')
    plt.show()

# Example usage:
# sgd_random_search(X_train, X_test, y_train, y_test)
