from flask import Flask, render_template, request
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load trained model
model = joblib.load("model.joblib")


df = pd.read_csv("bbc_news_dataset.csv")

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Text"])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    article = request.form["article"]

    prediction = model.predict([article])[0]

    same_category = df[df["Category"] == prediction]

    same_matrix = vectorizer.transform(same_category["Text"])

    article_vector = vectorizer.transform([article])

    similarity = cosine_similarity(article_vector, same_matrix)

    top3 = similarity.argsort()[0][-3:][::-1]

    recommendations = same_category.iloc[top3]["Text"].tolist()

    return render_template(
        "index.html",
        prediction=prediction,
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)