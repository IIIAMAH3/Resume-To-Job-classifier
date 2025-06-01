import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from functools import wraps
import os
import pickle

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

    df2 = pd.read_csv("UpdatedResumeDataSet.csv")[["Resume", "Category"]]
    df2 = df2.rename(columns={"Resume": "Resume_str"})

    main_df = pd.concat([df1, df2], ignore_index=True)

    main_df["Resume_str"] = main_df["Resume_str"].apply(lambda x: x.lower().strip())

    with open("preprocessed_resumes.pkl", "wb") as f:
        pickle.dump(main_df, f)

# Separate features and targets
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(main_df["Resume_str"])
y = main_df["Category"]

# Split the dataset into train/test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize features
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a classifier
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
