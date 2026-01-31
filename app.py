
from flask import Flask, render_template, request
import pickle
from preprocess import clean_text

app = Flask(__name__)
model, vectorizer = pickle.load(open("model/hate_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        text = request.form["tweet"]
        vec = vectorizer.transform([clean_text(text)])
        result = "Hate Speech" if model.predict(vec)[0] == 1 else "Non-Hate Speech"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
