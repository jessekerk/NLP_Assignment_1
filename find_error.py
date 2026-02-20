import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

"""
In this file i have copied some code from the other files, just to make 
things easier to me during the writing the code.
"""
# First i load the data
df = pd.read_json("hf://datasets/sh0416/ag_news/train.jsonl", lines=True)
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, shuffle=True, stratify=df["label"]
)
dev_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, shuffle=True, stratify=temp_df["label"]
)


def get_text_series(frame):
    return (
        frame["title"].fillna("") + " " + frame["description"].fillna("")
    ).str.strip()


# here we prepare the features
train_texts = get_text_series(train_df)
test_texts = get_text_series(test_df)
y_train = train_df["label"]
y_test = test_df["label"]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=2000,
    ngram_range=(1, 2),
    min_df=2,
)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# train the model
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# find out the misclassiefied examples
test_df["predicted"] = y_pred
errors = test_df[test_df["label"] != test_df["predicted"]].head(20)

# categorization
label_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

print(f"{'True Label':<10} | {'Predicted':<10} | {'Text Preview'}")
print("-" * 100)
for _, row in errors.iterrows():
    true_name = label_map[row["label"]]
    pred_name = label_map[row["predicted"]]
    text = f"{row['title']} - {row['description']}"[:80] + "..."
    print(f"{true_name:<10} | {pred_name:<10} | {text}")
