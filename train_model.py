
import pandas as pd, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import clean_text
import os
os.makedirs("model", exist_ok=True)

labeled = pd.read_csv("data/labeled.csv")
unlabeled = pd.read_csv("data/unlabeled.csv")

labeled['tweet'] = labeled['tweet'].apply(clean_text)
unlabeled['tweet'] = unlabeled['tweet'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=3000)
X_l = vectorizer.fit_transform(labeled['tweet'])
y_l = labeled['label']

base_model = LogisticRegression()
base_model.fit(X_l, y_l)

X_u = vectorizer.transform(unlabeled['tweet'])
probs = base_model.predict_proba(X_u)
preds = base_model.predict(X_u)

mask = probs.max(axis=1) >= 0.9

X_final = list(labeled['tweet']) + list(unlabeled['tweet'][mask])
y_final = list(y_l) + list(preds[mask])

X_final_vec = vectorizer.fit_transform(X_final)
final_model = LogisticRegression()
final_model.fit(X_final_vec, y_final)

pickle.dump((final_model, vectorizer), open("model/hate_model.pkl", "wb"))
print("Model trained & saved")
