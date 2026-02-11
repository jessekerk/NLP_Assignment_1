import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

splits = {"train": "train.jsonl", "test": "test.jsonl"}
df = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["train"], lines=True)

print(df.columns)
print(df.head())  # <-- note the ()

# train / (dev+test) temp split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# dev / test
dev_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, shuffle=True
)


def get_text_series(frame: pd.DataFrame) -> pd.Series:
    # best practice: combine title + description
    return (
        frame["title"].fillna("") + " " + frame["description"].fillna("")
    ).str.strip()


def get_tf_idf_preview(train_df: pd.DataFrame, n_rows: int = 5, n_cols: int = 30):
    texts = get_text_series(train_df)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
    )

    X = vectorizer.fit_transform(texts)

    # only densify a tiny slice for display
    names = vectorizer.get_feature_names_out()
    df_small = pd.DataFrame(X[:n_rows].toarray(), columns=names)

    # show only the most relevant columns so the printout is readable
    top_cols = df_small.sum(axis=0).sort_values(ascending=False).head(n_cols).index
    print(df_small[top_cols].round(3).to_string(index=False))

def get_tf_idf(train_df: pd.DataFrame):
    texts = get_text_series(train_df)
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=2000, ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(texts)
    names = vectorizer.get_feature_names_out()
    df_all = pd.DataFrame(X.toarray(), columns=names)
    print(df_all.round(3))
    
get_tf_idf(train_df)
