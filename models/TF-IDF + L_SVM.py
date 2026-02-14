import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from models.data import df


# Split train/dev/test
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, shuffle=True, stratify=df["label"]
)
dev_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, shuffle=True, stratify=temp_df["label"]
)


def get_text_series(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["title"].fillna("") + " " + frame["description"].fillna("")  # type: ignore
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

model = LinearSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_dev)
print("Dev accuracy:", accuracy_score(y_dev, y_pred))
print(classification_report(y_dev, y_pred))
print("confusion_matrix:", confusion_matrix(y_dev, y_pred))
