#%%
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import log_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

def xgboost_random_search(X_train, X_test, y_train, y_test):
    # Hyperparameter grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.5, 0.7, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3]
    }

    # Initialize XGBoost classifier
    xgb_clf = XGBClassifier(random_state=42)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        xgb_clf,
        param_distributions=param_dist,
        scoring='neg_log_loss',
        n_iter=10,  # Adjust the number of iterations as needed
        cv=3,
        n_jobs=-1,
        verbose=10
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

