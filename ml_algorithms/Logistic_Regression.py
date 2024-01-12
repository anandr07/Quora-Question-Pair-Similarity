#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def logistic_regression_function(X_train, X_test, y_train, y_test):
    # Initialize Logistic Regression model
    logreg_model = LogisticRegression(random_state=42)

    # Fit the model to the training data
    logreg_model.fit(X_train, y_train)

    # Predict probabilities for train and test sets
    train_probs = logreg_model.predict_proba(X_train)
    test_probs = logreg_model.predict_proba(X_test)

    # Calculate log loss for train and test sets
    train_log_loss = log_loss(y_train, train_probs, labels=logreg_model.classes_, eps=1e-15)
    test_log_loss = log_loss(y_test, test_probs, labels=logreg_model.classes_, eps=1e-15)

    # Display log loss for train and test sets
    print(f'Train Log Loss: {train_log_loss:.5f}')
    print(f'Test Log Loss: {test_log_loss:.5f}')

    # Predict labels for the test set
    predicted_labels = logreg_model.predict(X_test)

    # Display confusion matrix
    cm = confusion_matrix(y_test, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg_model.classes_)
    disp.plot(cmap='Blues', values_format='d')

    plt.title('Confusion Matrix')
    plt.show()

#%%