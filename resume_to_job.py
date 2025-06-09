import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time
from functools import wraps
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "lemmatizer"])

nlp.enable_pipe("lemmatizer")


# def timing_decorator(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time.perf_counter()  # high-precision timer
#         result = func(*args, **kwargs)
#         end_time = time.perf_counter()
#         duration = end_time - start_time
#         print(f"Function '{func.__name__}' took {duration:.4f} seconds to run.")
#         return result
#     return wrapper

if os.path.exists("preprocessed_resumes.pkl"):
    with open("preprocessed_resumes.pkl", "rb") as f:
        main_df = pickle.load(f)
else:
    df1 = pd.read_csv("Resume.csv")[["Resume_str", "Category"]]

    df2 = pd.read_csv("UpdatedResumeDataSet.csv")[["Resume_str", "Category"]]

    main_df = pd.concat([df1, df2], ignore_index=True)

    main_df["Resume_str"] = main_df["Resume_str"].apply(lambda x: x.lower().strip())
    main_df["Category"] = main_df["Category"].apply(lambda x: x.lower().strip())
    categories_to_drop = ["advocate", "agriculture", "apparel", "automobile", "bpo", "consultant", "arts", "aviation",
                          "banking", "business-development", "healthcare"]
    main_df = main_df[~main_df["Category"].isin(categories_to_drop)]

    with open("preprocessed_resumes.pkl", "wb") as f:
        pickle.dump(main_df, f)

# Separate features and targets
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(main_df["Resume_str"])
y = main_df["Category"]

# Split the dataset into train/test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a classifier
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

fig_height = max(12, len(report) * 0.4)
plt.figure(figsize=(10, fig_height))
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T,
            annot=True,
            cmap="Blues",
            fmt=".2f",
            cbar=True,
            linewidths=0.5,
            annot_kws={"size": 11})

plt.title("Classification Report", fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()
