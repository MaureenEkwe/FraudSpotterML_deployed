# 🔍 FraudSpotter README File
AI-Powered Job Posting Fraud Detection with Streamlit + Astra DB (Cassandra)

FraudSpotter is a machine learning application that detects potentially fraudulent job postings.

It integrates Streamlit, a trained TF-IDF + Logistic Regression model, and Astra DB (managed Apache Cassandra), enabling real-time predictions and database storage.

Created By: Maureen Ekwebelem and YaeJin(Sally) Kang

## 🚀 Features

### ✅ 1. Interactive Streamlit Web App
- Clean user interface for entering job posting details  
- Real-time scam likelihood prediction  
- Fraud probability visualization  

### ✅ 2. Machine Learning Model
- Pretrained TF-IDF vectorizer  
- Logistic Regression classifier trained on fraudulent job postings  
- Predicts probability that a posting is fake  

### ✅ 3. Cassandra Database Integration
Connects to Astra DB using the Data API and stores information including but not limited to:  
- job title  
- company  
- location  
- description  
- requirements  
- fraud probability  
- predicted label  
- timestamp 

---

## 📁 Project Structure
files
- app.py
- tfidf_vectorizer.pkl
- fraud_spotter.pkl
- README.md
- logo.png
- training model.py (optional- shows how ML classifier was trained)
---

## 🧠 How It Works

### **1. Input**
User provides job information (title, company, description, requirements, etc.)

### **2. ML Prediction**
Text is cleaned → vectorized (TF-IDF) → Logistic Regression predicts:  
- **fraud_probability**  
- **predicted_label**  

### **3. Save to Cassandra**
Record inserted into:  
`default_keyspace.job_postings`

---

💻 Running the App (Colab)
1. Install dependencies
pip install streamlit pyngrok astrapy joblib scikit-learn
* Note: make sure to install the latest version of scikit-learn 1.6.1 

2. Run Streamlit
!streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false &

3. Start ngrok tunnel
from pyngrok import ngrok
ngrok.kill()
public_url = ngrok.connect(8501)
public_url


Use the generated URL to access the app.

🔐 Environment Variables
ASTRA_TOKEN = "AstraCS:..." 
ASTRA_ENDPOINT = "https://<db-id>-<region>.apps.astra.datastax.com"
