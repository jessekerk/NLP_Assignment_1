import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from data.data import train_df, dev_df, test_df


def run_tfidf_lsvm():
    """Runs the TF-IDF + LSVM model"""
    def get_text_series(frame: pd.DataFrame) -> pd.Series:
        """Gets title + description as a data point (Missing vals are replaces w/ empty strings)

        Args:
            frame (pd.DataFrame): dataframe with column title and description

        Returns:
            pd.Series: the concatenated, cleaned data consisting of title and description
        """
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

    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_dev)
    return {
        "accuracy": accuracy_score(y_dev, y_pred),
        "report": classification_report(y_dev, y_pred),
        "confusion_matrix": confusion_matrix(y_dev, y_pred),
    }
