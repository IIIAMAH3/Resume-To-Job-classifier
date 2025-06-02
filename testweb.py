import pandas as pd
import spacy
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle
import os
from PyPDF2 import PdfReader
from io import BytesIO

# Load spaCy model
nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "lemmatizer"])
nlp.enable_pipe("lemmatizer")

# Page configuration
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Cache the model loading to avoid reloading on every interaction
@st.cache_resource(show_spinner="Loading classification model...")
def load_model():
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

    # Train model
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(main_df["Resume_str"])
    y = main_df["Category"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)

    # Return all necessary components
    return vectorizer, scaler, clf


# PDF text extraction function
def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# Prediction function
def predict_resume(resume_text, vectorizer, scaler, model):
    # Preprocess input text
    processed_text = resume_text.lower().strip()

    # Transform using trained vectorizer and scaler
    X = vectorizer.transform([processed_text])
    X_scaled = scaler.transform(X)

    # Make prediction
    return model.predict(X_scaled)[0]


# Main application
def main():
    st.title("📄 AI Resume Classifier")
    st.markdown("Upload a resume to predict its job category")

    # Load model components
    vectorizer, scaler, model = load_model()

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Resume", "Paste Text"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload a resume (PDF or TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=False,
            key="file_uploader"
        )

        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                resume_text = extract_pdf_text(uploaded_file)
            else:  # text file
                resume_text = uploaded_file.getvalue().decode("utf-8")

            with st.expander("View Resume Text"):
                st.text(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""))

            if st.button("Classify Resume", key="classify_file"):
                with st.spinner("Analyzing resume..."):
                    prediction = predict_resume(resume_text, vectorizer, scaler, model)
                    st.success(f"**Predicted Category:** {prediction}")
                    st.balloons()

    with tab2:
        resume_text = st.text_area(
            "Paste resume content here:",
            height=300,
            placeholder="Paste resume text content here...",
            key="text_area"
        )

        if st.button("Classify Resume", key="classify_text"):
            if not resume_text.strip():
                st.warning("Please enter resume content")
            else:
                with st.spinner("Analyzing resume..."):
                    prediction = predict_resume(resume_text, vectorizer, scaler, model)
                    st.success(f"**Predicted Category:** {prediction}")
                    st.balloons()

    # Model information in sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This AI-powered tool classifies resumes into job categories using machine learning.
        - **Model:** Support Vector Machine (SVM)
        - **Features:** TF-IDF Vectorization
        - **Training Data:** 2,500+ resumes across 25 categories
        """)

        st.header("How to Use")
        st.markdown("""
        1. Upload a PDF or TXT resume
        2. Or paste resume text directly
        3. Click 'Classify Resume'
        4. View predicted job category
        """)

        st.header("Categories")
        st.markdown("""
        - Data Science
        - HR
        - Designer
        - Information Technology
        - Teacher
        - Business Analyst
        - Civil Engineer
        - Web Developer
        - Mechanical Engineer
        - Sales
        - Operations
        - Electrical Engineering
        - Banking
        - Marketing
        - Health and Fitness
        - PMO
        - Retail
        - Arts
        - Food and Beverages
        - Petroleum
        - Media
        - Aviation
        - Legal
        - Consulting
        - Digital Media
        """)


if __name__ == "__main__":
    main()