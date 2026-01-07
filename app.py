
# ==========================
import os
import re
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from astrapy import DataAPIClient

# ==========================
# STREAMLIT PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="FraudSpotter",
    page_icon="🔍",
    layout="wide"
)

# ==========================
# ASTRA CONFIG
# ==========================
ASTRA_TOKEN = os.getenv("ASTRA_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_ENDPOINT")
ASTRA_COLLECTION_NAME = os.getenv("ASTRA_COLLECTION_NAME", "fraud_predictions")


# ==========================
# LOAD MODEL + VECTORIZER
# ==========================
@st.cache_resource
def load_artifacts():
    """
    Load TF-IDF vectorizer and trained model from local files.
    Assumes they are in the same folder as this app.py.
    """
    base_dir = os.path.dirname(__file__)
    vectorizer_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")
    model_path = os.path.join(base_dir, "fraud_spotter.pkl")  # use your actual filename

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model


vectorizer, model = load_artifacts()


# ==========================
# CONNECT TO ASTRA (DATA API)
# ==========================
@st.cache_resource
def get_job_collection():
    client = DataAPIClient(ASTRA_TOKEN)
    db = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)
    return db.get_collection(ASTRA_COLLECTION_NAME)


job_collection = get_job_collection()


# ==========================
# TEXT CLEANING + PREDICTION
# ==========================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_fake(full_text: str):
    cleaned = clean_text(full_text)
    X_input = vectorizer.transform([cleaned])
    prob = model.predict_proba(X_input)[0][1]
    label = prob >= 0.5  # True = fake, False = real
    return prob, label


def save_job_to_db(
    title,
    company,
    location,
    description,
    requirements,
    benefits,
    employment_type,
    required_experience,
    required_education,
    industry,
    function,
    salary_range,
    prob,
    label,
):
    doc = {
        "title": title,
        "company": company,
        "location": location,
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": function,
        "salary_range": salary_range,
        "fraud_probability": float(prob),
        "predicted_label": bool(label),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    job_collection.insert_one(doc)
    return doc


# ==========================
# CUSTOM CSS
# ==========================
st.markdown(
    """
    <style>
        .main {
            background-color: #0b1220;
            color: #f9fafb;
        }
        .block-title {
            font-size: 35px;
            font-weight: 700;
            color: #111827;
        }
        .subtext {
            color: #9ca3af;
        }
        .result-box {
            padding: 1.2rem;
            border-radius: 0.75rem;
            border: 1px solid #1f2937;
            background: #020617;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# SIDEBAR NAVIGATION
# ==========================
#st.sidebar.title("🔍 FraudSpotter")

st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.caption("Job scam detector")


page = st.sidebar.radio(
    "Navigation",
    ["🔮 Predict", "🔒  Analytics (Admin Only)", "ℹ️ About"],
)

# ==========================
# PAGE 1 – PREDICT
# ==========================
if page == "🔮 Predict":
    # HEADER
    st.markdown('<div class="block-title">🔍 FraudSpotter</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtext">Paste a job posting below and our model will estimate how likely it is to be fraudulent.</p>',
        unsafe_allow_html=True,
    )

    st.write("---")

    # LAYOUT: LEFT = INPUTS, RIGHT = RESULT
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Job information")

        with st.container():
            st.markdown("**Basics**")
            title = st.text_input("Job title *")
            company = st.text_input("Company (optional)")
            location = st.text_input("Location (optional)")

        st.markdown("**Details**")
        description = st.text_area("Job description *", height=160)
        requirements = st.text_area("Requirements (optional)", height=120)
        benefits = st.text_area("Benefits (optional)", height=120)

        col1, col2 = st.columns(2)
        with col1:
            employment_type = st.text_input("Employment type (optional)")
            required_experience = st.text_input("Required experience (optional)")
        with col2:
            required_education = st.text_input("Required education (optional)")
            industry = st.text_input("Industry (optional)")

        function = st.text_input("Function / role category (optional)")
        salary_range = st.text_input("Salary range (optional)")

        analyze = st.button("✨ Analyze posting", use_container_width=True)

    with right:
        st.subheader("Prediction")

        if "last_result" not in st.session_state:
            st.session_state.last_result = None

        if analyze:
            parts = [
                title,
                description,
                requirements,
                company,
                benefits,
                employment_type,
                required_experience,
                required_education,
                industry,
                function,
                location,
                salary_range,
            ]
            full_text = " ".join([p for p in parts if p])

            if not title or not description:
                st.warning("Please fill in at least the job title and job description.")
            else:
                prob, label = predict_fake(full_text)

                # Save to Astra
                try:
                    save_job_to_db(
                        title=title,
                        company=company,
                        location=location,
                        description=description,
                        requirements=requirements,
                        benefits=benefits,
                        employment_type=employment_type,
                        required_experience=required_experience,
                        required_education=required_education,
                        industry=industry,
                        function=function,
                        salary_range=salary_range,
                        prob=prob,
                        label=label,
                    )
                    st.success("Saved to database ✅")
                except Exception as e:
                    st.warning(f"Could not save to database: {e}")

                st.session_state.last_result = (prob, label, full_text)

        if st.session_state.last_result is not None:
            prob, label, full_text = st.session_state.last_result

            if label:
                st.markdown("#### ⚠️ Likely **FAKE** job posting")
                st.markdown(
                    "This posting shares patterns with known fraudulent ads. Proceed with caution."
                )
            else:
                st.markdown("#### ✅ Likely **REAL** job posting")
                st.markdown(
                    "This posting looks similar to legitimate job ads in our training data."
                )

            st.metric("Fraud Probability", f"{prob * 100:.1f} %")
            st.progress(min(max(prob, 0.0), 1.0))

        else:
            st.info("Prediction will appear here after you click **Analyze posting**.")


# ==========================
# PAGE 2 – Analytics
# ==========================
if page == "🔒  Analytics (Admin Only)":
    st.markdown("## 🔒 Prediction History/Analytics (Admin Only)")
    st.write("Recent job postings and model predictions stored in Astra DB for Analytics Purposes.")

    try:
        docs = list(job_collection.find({}))
    except Exception as e:
        st.error(f"Could not read from database: {e}")
        docs = []

    if not docs:
        st.info("No predictions saved yet. Go to **Predict** and analyze a posting first.")
    else:
        df = pd.DataFrame(docs)

        # drop internal _id if present
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])

        cols_to_show = [
            c
            for c in [
                "created_at",
                "title",
                "company",
                "location",
                "fraud_probability",
                "predicted_label",
            ]
            if c in df.columns
        ]

        st.subheader("Saved predictions")
        st.dataframe(df[cols_to_show])

        st.subheader("Summary")
        col1, col2 = st.columns(2)

        with col1:
            if "predicted_label" in df.columns:
                counts = df["predicted_label"].value_counts()
                if not counts.empty:
                    counts.index = counts.index.map({True: "Fraud", False: "Legit"})
                    st.write("Count of predictions")
                    st.bar_chart(counts)

        with col2:
            if "fraud_probability" in df.columns:
                st.write("Fraud probability over saved jobs")
                st.line_chart(df["fraud_probability"])


# ==========================
# PAGE 3 – ABOUT
# ==========================
if page == "ℹ️ About":
    st.markdown("## ℹ️ About FraudSpotter")
    st.write(
        """
        FraudSpotter is a job scam detection tool built with:

        - 🧠 A machine learning model (TF-IDF + classifier)
        - 🗃️ Astra DB (NoSQL) to store job postings and predictions
        - 🎨 Streamlit for the interactive web app

        Use the **Predict** tab to analyze a job posting, and the **History** tab
        to review previous predictions stored in the database.
        """
    )
    st.markdown("Built by FraudSpotter 🔍")
