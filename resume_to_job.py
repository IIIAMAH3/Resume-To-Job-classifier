import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import re
import time
from functools import wraps


nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "lemmatizer"])

nlp.enable_pipe("lemmatizer")

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # high-precision timer
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Function '{func.__name__}' took {duration:.4f} seconds to run.")
        return result
    return wrapper


@timing_decorator
def preprocess_text(text):
    # Remove special characters/numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Lowercase and lemmatize
    doc = nlp(text.lower().strip())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and len(token.lemma_) > 2
    ]
    return ''.join(tokens)

df1 = pd.read_csv("Resume.csv")[["Resume_str", "Category"]]

df2 = pd.read_csv("UpdatedResumeDataSet.csv")[["Resume", "Category"]]
df2 = df2.rename(columns={"Resume": "Resume_str"})

# The file was originally encoded in a different format and then saved incorrectly as UTF-8
df2["Resume_str"] = df2["Resume_str"].str.encode("latin1").str.decode("utf-8")

main_df = pd.concat([df1, df2], ignore_index=True)
# Store it in a pickle file
# main_df["Resume_str"] = main_df["Resume_str"].apply(preprocess_text)
print(4)
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

# resume = """Motivated and skilled Software Developer with hands-on experience in building dynamic web
# applications and backend services using Python and Django. Proficient in database
# management with SQL and PostgreSQL, along with version control using Git. Adept at
# developing custom scripts and deploying scalable solutions. Proven track record of
# delivering high-quality code and maintaining a personal web portfolio showcasing diverse
# projects.
# Skills
# ● Programming Languages: Python
# ● Web Development: Django, HTML, CSS, JavaScript
# ● Databases: SQL, PostgreSQL
# ● Version Control: Git
# ● Soft Skills: Leadership, problem-solving, teamwork, self-motivation, adaptability"""
# vectorizer2 = TfidfVectorizer()
# resume = vectorizer2.fit_transform([resume])
# scaler2 = StandardScaler(with_mean=False)
# resume_scaled = scaler2.fit_transform(resume)
#
# y_test_pred = clf.predict(resume)
# print(y_test_pred)