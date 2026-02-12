import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# load data
df = pd.read_json("hf://datasets/sh0416/ag_news/train.jsonl", lines=True)


# Split train/dev/test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
dev_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, shuffle=True
)


def get_text_series(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["title"].fillna("") + " " + frame["description"].fillna("") # type: ignore
    ).str.strip()


# Text + labels
train_texts = get_text_series(train_df)
dev_texts = get_text_series(dev_df)
test_texts = get_text_series(test_df)

y_train = train_df["label"]
y_dev = dev_df["label"]
y_test = test_df["label"]

# TF-IDF (fit on train only)
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=2000,
    ngram_range=(1, 2),
    min_df=2,
)

X_train = vectorizer.fit_transform(train_texts)
X_dev = vectorizer.transform(dev_texts)
X_test = vectorizer.transform(test_texts)

print("X_train shape:", X_train.shape)
print("X_dev shape:", X_dev.shape)

# Logistic Regression
model = LogisticRegression(max_iter=1000, solver="lbfgs")
model.fit(X_train, y_train)

# Evaluate on dev
y_pred = model.predict(X_dev)
print("Dev accuracy:", accuracy_score(y_dev, y_pred))
