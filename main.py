import numpy as np
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def read_data(folder_path):
    reviews = []
    for file_name in os.listdir(folder_path):
        file_name_path = os.path.join(folder_path, file_name)
        with open(file_name_path, 'r', encoding='utf-8') as f:
            reviews.append(f.read())
    return reviews

def load_data(positive_folder, negative_folder):
    positive = read_data(positive_folder)
    negative = read_data(negative_folder)
    return positive, negative


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    report  = classification_report(y_test, prediction)
    cm = confusion_matrix(y_test, prediction)
    return accuracy, report, cm

pwd = os.getcwd()
positive_reviews_folder = os.path.join(pwd, 'aclImdb', 'train', 'pos')
negative_reviews_folder = os.path.join(pwd, 'aclImdb', 'train', 'neg')
positive_test_folder = os.path.join(pwd, 'aclImdb', 'test', 'pos')
negative_test_folder = os.path.join(pwd, 'aclImdb', 'test', 'neg')

positive_reviews, negative_reviews = load_data(positive_reviews_folder, negative_reviews_folder)
positive_test, negative_test = load_data(positive_test_folder, negative_test_folder)

X = np.array(positive_reviews + negative_reviews)
y = np.array([0]*len(positive_reviews)+[1]*len(negative_reviews))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X
y_train = y

X = np.array(positive_test + negative_test)
y = np.array([0]*len(positive_test)+[1]*len(negative_test))

X_test = X
y_test = y

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

models = {
    "Gradient Boosting": GradientBoostingClassifier(),
    "MultinomialNB": MultinomialNB(),
    "Random forest": RandomForestClassifier(),
    "Logistic regression": LogisticRegression(max_iter = 1000),
    "SVC": SVC(kernel = 'linear')
}

for name, model in models.items():
    print(f"Evaluating model: {name}")
    accuracy, report, cm = evaluate_model(model, X_train_tfidf, y_train, X_test_tfidf, y_test)
    print(f"Accuracy: {accuracy}")
    print("Report: ")
    print(report)
    print("Confusion matrix: ")
    print(cm)

# Grid search tested for Logistic Regression didn't show any significant change in accuracy
#param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
#grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
#grid_search.fit(X_train_tfidf, y_train)
#print("Best parameters for Logistic Regression: ", grid_search.best_params_)