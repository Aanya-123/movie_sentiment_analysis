import pandas as pd, re, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

df = pd.read_csv("IMDB Dataset.csv")          # columns: review, sentiment
df["review"] = df["review"].apply(lambda t: re.sub(r"<.*?>", " ", t).lower())
df["label"] = (df["sentiment"] == "positive").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                              max_features=100_000, min_df=3)),
    ("clf", LogisticRegression(max_iter=1000, C=2.0)),
])
pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))   # expect ~89-90% accuracy

joblib.dump(pipe, "sentiment_model.joblib") 