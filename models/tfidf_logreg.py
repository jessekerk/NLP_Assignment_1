import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from data.data import train_df, dev_df


def run_tfidf_logreg():
    def get_text_series(frame: pd.DataFrame) -> pd.Series:
        """Gets title + description as a data point (Missing vals are replaces w/ empty strings)

        Args:
            frame (pd.DataFrame): dataframe with column title and description

        Returns:
            pd.Series: the concatenated, cleaned data consisting of title and description
        """
        return (
            frame["title"].fillna("") + " " + frame["description"].fillna("")
        ).str.strip()

    # texts
    train_texts = get_text_series(train_df)
    dev_texts = get_text_series(dev_df)

    y_train = train_df["label"]
    y_dev = dev_df["label"]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=2000,
        ngram_range=(1, 2),
        min_df=2,
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_dev = vectorizer.transform(dev_texts)

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_dev)

    return {
        "accuracy": accuracy_score(y_dev, y_pred),
        "report": classification_report(y_dev, y_pred),
        "confusion_matrix": confusion_matrix(y_dev, y_pred),
    }
