from flask import Flask, render_template, request
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import spacy
import csv
import pandas as pd
from sklearn.metrics import recall_score


app = Flask(__name__)

# load a pre-trained English language model
nlp = spacy.load("en_core_web_sm")

# define a list of idioms and their original meanings

idioms =[]
with open('english_idioms.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        idiom = row['idioms']
        meaning = row['meaning']
        idioms.append((idiom, meaning))

# preprocess the idioms by removing stop words, punctuation, and converting to lowercase
def preprocess(text):
    text = re.sub(r"[^\w\s]", "", text)
    doc = nlp(text.lower())
    return " ".join([token.text for token in doc if not token.is_stop])

# extract the idioms and their meanings into separate lists
X, y = zip(*idioms)

# preprocess the idioms
X = [preprocess(text) for text in X]

# create a TfidfVectorizer to convert the idioms to numerical features
vectorizer = TfidfVectorizer()

# transform the idioms into feature vectors
X_vectors = vectorizer.fit_transform(X)

# create a LinearSVC model to predict the meanings
model = LinearSVC()

# fit the model to the data
model.fit(X_vectors, y)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    # get the user's input from the form
    idiom = request.form["idiom"]

    # preprocess the user's input
    idiom = preprocess(idiom)

    # transform the user's input into a feature vector
    idiom_vector = vectorizer.transform([idiom])

    # use the model to predict the idiom's original meaning
    prediction = model.predict(idiom_vector)

    # render the translation template with the predicted meaning
    return render_template("result.html", idiom=idiom, prediction=prediction[0])
    


@app.route("/evaluate")
def evaluate():
    # get all possible meanings for the idioms
    all_meanings = set([meaning for _, meaning in idioms])

    # initialize a dictionary to store the precision for each meaning
    precision_dict = {meaning: 0 for meaning in all_meanings}

    # preprocess the idioms and create a list of dictionaries to store the results
    results = []
    for idiom, meaning in idioms:
        X = preprocess(idiom)
        X_vector = vectorizer.transform([X])
        y_pred = model.predict(X_vector)[0]
        result = {
            "idiom": idiom,
            "true_meaning": meaning,
            "predicted_meaning": y_pred,
            "correct_prediction": meaning == y_pred
        }
        results.append(result)

        # update the precision for the predicted meaning
        precision_dict[y_pred] += int(meaning == y_pred)

    # calculate the accuracy, precision, and recall
    accuracy = accuracy_score([result["true_meaning"] for result in results], [result["predicted_meaning"] for result in results])
    precision = sum(precision_dict.values()) / len(results)
    recall = recall_score([result["true_meaning"] for result in results], [result["predicted_meaning"] for result in results], average="macro")

    # create a confusion matrix
    confusion_matrix = pd.crosstab([result["true_meaning"] for result in results], [result["predicted_meaning"] for result in results])

    # render the evaluation template with the performance metrics
    return render_template("result.html", accuracy=accuracy*100, precision=precision*100, recall=recall*100)


if __name__ == "__main__":
    app.run(debug=True)
