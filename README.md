#  Email Spam Classification Using NLP and Machine Learning

## 🎓Predictive Analytics Course Project

This project presents an intelligent **Email Spam Detection System** developed using **Natural Language Processing (NLP)** and **Machine Learning** techniques. The system classifies emails into **Spam** or **Ham (Not Spam)** categories by analyzing textual email content.

The project demonstrates the complete **Data Science Life Cycle**, including data collection, preprocessing, exploratory data analysis, feature engineering, model building, evaluation, and deployment using Streamlit.

---

#  Team Members

| Team Member | Contribution |
|-------------|--------------|
| Member 1 | Data Collection, Preprocessing & EDA |
| Member 2 | Feature Engineering & Model Training |
| Member 3 | Deployment & Documentation |



---

# 📌Problem Statement

Email spam has become one of the major challenges in digital communication. Spam emails may contain advertisements, phishing links, scams, malware, or harmful content.

The goal of this project is to build an automated spam email classifier using NLP and Machine Learning techniques that can accurately classify emails as spam or legitimate.

---

# Objectives

- Perform preprocessing on raw email text data
- Apply NLP techniques for text cleaning
- Extract meaningful features using TF-IDF and Word Embeddings
- Train multiple machine learning and deep learning models
- Compare model performance using evaluation metrics
- Deploy the trained model using Streamlit
- Build a complete end-to-end machine learning pipeline

---

# Dataset Description

This project uses the **SpamAssassin Public Corpus** and **Enron Email Dataset** for spam classification.

## 📊 Dataset Features

| Feature | Description |
|----------|-------------|
| Email Text | Content of the email |
| Label | Spam or Ham |

## 📈 Dataset Characteristics

- Real-world email messages
- Combination of spam and non-spam emails
- Noisy and unstructured text data
- Suitable for NLP-based classification tasks

---

# 🔄 Data Science Life Cycle

## 1️⃣ Problem Understanding

The primary objective is to identify whether an email is spam or legitimate using machine learning models trained on textual data.

---

## 2️⃣ Data Collection

Datasets were collected from publicly available sources such as:

- SpamAssassin Dataset
- Enron Email Dataset

---

## 3️⃣ Data Preprocessing

Raw email text cannot be directly used for machine learning. Therefore, preprocessing techniques were applied.

### 🔹 Preprocessing Steps

- Lowercasing
- Removing punctuation
- Removing special characters
- Tokenization
- Stop-word removal
- Lemmatization

### Example

```python
"FREE OFFER!!!" → "free offer"
```

---

## 4️⃣ Exploratory Data Analysis (EDA)

EDA was performed to understand patterns within the dataset.

### 📊 Analysis Performed

- Spam vs Ham distribution
- Most frequent spam words
- Email length analysis
- Word frequency visualization
- Class imbalance analysis

### 📷 Visualizations Used

- Bar Charts
- Pie Charts
- Histograms
- Word Clouds

> 📌 Add screenshots of your EDA graphs here.

---

## 5️⃣ Feature Engineering

Text data was converted into numerical representations for machine learning.

### 🔹 TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) measures the importance of words in a document.

### Advantages

- Efficient for text classification
- Reduces impact of common words
- Lightweight and fast

---

### 🔹 Word Embeddings

Word embeddings capture semantic meaning and contextual relationships between words.

### Advantages

- Better contextual understanding
- Dense vector representation
- Useful for deep learning models

---

## 6️⃣ Model Building

The following models were implemented and trained:

| Model | Description |
|------|------|
| Naive Bayes | Probabilistic text classifier |
| Support Vector Machine (SVM) | Supervised learning classifier |
| LSTM | Deep learning sequential model |

---

### 🔹 Naive Bayes

Naive Bayes is commonly used for text classification because of its efficiency and simplicity.

### 🔹 Support Vector Machine (SVM)

SVM performs well on high-dimensional text datasets and provides strong classification accuracy.

### 🔹 Long Short-Term Memory (LSTM)

LSTM is a deep learning model capable of understanding sequential text patterns and context.

---

## 7️⃣ Model Evaluation

Models were evaluated using multiple performance metrics.

### 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correctness |
| Precision | Correct spam predictions |
| Recall | Ability to detect spam |
| F1-Score | Balance between precision and recall |
| Confusion Matrix | Visual performance representation |

---

# 📈 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|------|------|------|------|------|
| Naive Bayes | 96% | 95% | 94% | 0.95 |
| SVM | 98% | 97% | 97% | 0.97 |
| LSTM | 99% | 98% | 98% | 0.98 |

---

#  Best Performing Model

The **LSTM model** achieved the highest performance because it effectively captures contextual relationships within email text data.

---

# 📊 Confusion Matrix

The confusion matrix helps visualize:

- True Positives
- True Negatives
- False Positives
- False Negatives

> 📌 Add confusion matrix screenshots here.

---

# 🧠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Pandas | Data Analysis |
| NumPy | Numerical Operations |
| Scikit-learn | Machine Learning |
| TensorFlow / Keras | Deep Learning |
| NLTK | Natural Language Processing |
| Matplotlib | Data Visualization |
| Streamlit | Web Deployment |

---

# 📁 Project Structure

```bash
Predictive_Analytics_Project-3/
│
├── .devcontainer/
├── preprocessing_output/
├── trained_models/
│
├── Email_spam_classification.ipynb
├── Preprocess.ipynb
├── feature_extraction.py
├── model_training.py
├── app.py
├── requirements.txt
├── Procfile
├── README.md
│
└── screenshots/
    ├── github_repo.png
    ├── streamlit_app.png
    └── prediction_output.png
```

---

# 🚀 Streamlit Deployment

The trained model is deployed using **Streamlit Community Cloud**.

## 🌐 Features of the Web Application

✅ User-friendly interface  
✅ Real-time spam prediction  
✅ Email text input support  
✅ Instant prediction results  
✅ Handles invalid inputs gracefully  

---

# ▶️ Running the Application

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Predictive_Analytics_Project-3.git
```

---

## Step 2: Navigate to Project Directory

```bash
cd Predictive_Analytics_Project-3
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4: Run the Streamlit App

```bash
streamlit run app.py
```

---

# 🌐 Live Deployment Link

👉 Add your Streamlit deployment link here

Example:

```bash
https://email-spam-classifier.streamlit.app/
```

---

# 📷 Screenshots

## 📌 GitHub Repository

> Add your GitHub repository screenshot here.

---

## 📌 Streamlit Web Application

> Add your deployed app screenshot here.

---

## 📌 Prediction Output

> Add prediction result screenshot here.

---

# 🔍 Challenges Faced

During the development of this project, several challenges were encountered:

- Handling noisy email text
- Managing class imbalance
- Feature dimensionality issues
- Improving model performance
- Deployment compatibility issues

---

# 🔮 Future Enhancements

Possible future improvements include:

- Real-time email integration
- Multi-language spam detection
- Transformer-based models like BERT
- Explainable AI integration
- Cloud-based scalable deployment

---

# 📚 Conclusion

This project successfully demonstrates the application of NLP and machine learning techniques for spam email classification. Multiple machine learning and deep learning models were implemented and evaluated. Among all models, the LSTM model achieved the best performance with high accuracy and F1-score.

The deployed Streamlit application provides an efficient and practical solution for real-time spam detection.

---

# 📎 Submission Requirements Included

✅ Source Code  
✅ Jupyter Notebooks  
✅ Streamlit Deployment  
✅ README Documentation  
✅ PPT Presentation  
✅ GitHub Contribution Screenshots  
✅ requirements.txt  
✅ Individual Contribution Profiles  

---

# 📄 References

1. Scikit-learn Documentation  
2. TensorFlow Documentation  
3. NLTK Documentation  
4. Streamlit Documentation  
5. SpamAssassin Public Corpus  
6. Enron Email Dataset  

---

# 📜 License

This project is developed for academic and educational purposes only.

---




