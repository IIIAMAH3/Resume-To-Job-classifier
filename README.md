For resume_to_job.py:
# Job Title Classification using TF-IDF and Support Vector Classifier (SVC)

This project builds a machine learning model that classifies job titles or descriptions into specific categories using natural language processing and supervised learning. It utilizes TF-IDF for feature extraction and an SVC (Support Vector Classifier) for classification.

---

## 📌 Features

- Text classification using job descriptions
- TF-IDF vectorization of text data
- Multi-class classification using Support Vector Classifier
- Evaluation using:
  - Precision, Recall, and F1-score
  - Confusion matrix
  - Heatmap visualization of the classification report

---

## 🧠 Model Workflow

1. **Preprocessing**: Clean and prepare textual job data.
2. **Feature Extraction**: Convert text to numerical features using `TfidfVectorizer`.
3. **Model Training**: Train a Support Vector Classifier (`sklearn.svm.SVC`) on TF-IDF vectors.
4. **Evaluation**: 
   - Visualize precision, recall, F1-score via seaborn heatmap
   - Show misclassifications using confusion matrix

---


---

## 🧪 Example Outputs

### 📊 Classification Report Heatmap
- Shows per-class metrics for precision, recall, and F1-score.
- Useful to detect weak categories.

### 🧩 Confusion Matrix
- Shows which classes are confused with others.
- Helps improve misclassified classes.

---

## 📦 Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

For testweb.py:

# Resume Classification Web Application

## Overview

This web application uses machine learning to classify resumes into 25 different job categories. Built with Python and Streamlit, it leverages a Support Vector Machine (SVM) classifier trained on over 2,500 resumes across various professional domains. The application extracts text from uploaded resumes or accepts direct text input, processes it using NLP techniques, and predicts the most suitable job category.

## Key Features

- **PDF & Text Support**: Upload PDF resumes or paste text directly
- **Real-time Prediction**: Get instant category predictions
- **25 Job Categories**: From Data Science to Aviation and Legal
- **Machine Learning Model**: SVM classifier with TF-IDF feature extraction
- **User-Friendly Interface**: Clean, intuitive design with visual feedback
- **Responsive Design**: Works on desktop and mobile devices

## Categories Supported

The model classifies resumes into the following 25 job categories:

1. Data Science
2. HR
3. Designer
4. Information Technology
5. Teacher
6. Business Analyst
7. Civil Engineer
8. Web Developer
9. Mechanical Engineer
10. Sales
11. Operations
12. Electrical Engineering
13. Banking
14. Marketing
15. Health and Fitness
16. PMO
17. Retail
18. Arts
19. Food and Beverages
20. Petroleum
21. Media
22. Aviation
23. Legal
24. Consulting
25. Digital Media

## Installation

Follow these steps to set up the application on your local machine:

### Prerequisites

- Python 3.8+
- pip package manager

### Step-by-Step Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/resume-classifier.git
   cd resume-classifier
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_md
   ```

4. **Download datasets**:
   - Place `Resume.csv` and `UpdatedResumeDataSet.csv` in the project root
   - (Optional) Preprocessed data will be automatically generated on first run

## Usage

### Running the Application

Start the application with:
```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`

### Using the Interface

1. **Upload a Resume**:
   - Click on the "Upload Resume" tab
   - Upload a PDF or TXT file
   - Click "Classify Resume" to get prediction

2. **Paste Resume Text**:
   - Click on the "Paste Text" tab
   - Enter resume content in the text area
   - Click "Classify Resume" to get prediction

3. **View Results**:
   - Predicted category appears in a success message
   - View additional information in the sidebar

### Sample Resumes

Test the application with these sample phrases:

- Data Science: "Machine learning, Python, TensorFlow, data analysis, statistical modeling"
- Web Developer: "JavaScript, React, Node.js, HTML5, CSS3, REST APIs, frontend development"
- HR: "Talent acquisition, employee relations, recruitment strategies, onboarding processes"

## Project Structure

```
resume-classifier/
├── app.py                 # Main application code
├── requirements.txt       # Dependencies
├── Resume.csv             # Dataset 1
├── UpdatedResumeDataSet.csv # Dataset 2
├── preprocessed_resumes.pkl # Preprocessed data (auto-generated)
├── README.md              # This file
└── .gitignore             # Ignore data files and virtual environments
```

## Dependencies

The application uses the following Python packages:

- streamlit==1.30.0
- pandas==2.1.1
- scikit-learn==1.3.2
- spacy==3.7.2
- PyPDF2==3.0.1
- joblib==1.3.2

All dependencies are listed in `requirements.txt`

## Customization

You can modify the application by:

1. **Training Data**:
   - Add more resumes to the CSV files
   - Include new categories by adding labeled examples

2. **Model**:
   - Edit `app.py` to use a different classifier
   - Adjust TF-IDF parameters in the vectorizer

3. **Interface**:
   - Modify Streamlit components in `app.py`
   - Add visualizations like confidence scores

## Troubleshooting

Common issues and solutions:

- **Model not loading**: Ensure spaCy model is installed with `python -m spacy download en_core_web_md`
- **File encoding issues**: Check CSV files are properly formatted
- **Slow performance**: First run takes longer to preprocess data
- **Prediction errors**: Ensure resumes contain sufficient relevant text

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contribution

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

---

**Note**: For production deployment, consider using Streamlit Cloud, Heroku, or AWS for hosting.
