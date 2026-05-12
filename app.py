"""
app.py — Email Spam Classification Streamlit App
Project #3: Email Spam Classification Using NLP and Machine Learning
"""

import os
import re
import string
import pickle
import warnings
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    TF_AVAILABLE = True
except (ImportError, Exception):
    TF_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    for resource in ["stopwords", "punkt"]:
        try:
            nltk.data.find(f"corpora/{resource}" if resource == "stopwords" else f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    NLTK_AVAILABLE = True
    _STOP_WORDS = set(stopwords.words("english"))
    _STEMMER = PorterStemmer()
except (ImportError, Exception):
    NLTK_AVAILABLE = False
    _STOP_WORDS = set()
    _STEMMER = None

# ── Constants ─────────────────────────────────────────────────────────────────
# Supports BOTH layouts:
#   Layout A: model files sit next to app.py in the repo root
#   Layout B: model files are inside a trained_models/ subfolder
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_SUBDIR   = os.path.join(ROOT_DIR, "trained_models")
MODEL_DIR = _SUBDIR if os.path.isdir(_SUBDIR) else ROOT_DIR

MODEL_FILES = {
    "Naive Bayes": os.path.join(MODEL_DIR, "naive_bayes_model.pkl"),
    "SVM":         os.path.join(MODEL_DIR, "svm_model.pkl"),
    "LSTM":        os.path.join(MODEL_DIR, "lstm_model.h5"),
    "tokenizer":   os.path.join(MODEL_DIR, "lstm_tokenizer.pkl"),
    "lstm_config": os.path.join(MODEL_DIR, "lstm_config.pkl"),
    "tfidf":       os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
}

EXAMPLE_EMAILS = {
    "🚨 Spam: Prize Winner":
        "Congratulations! You have been selected as the WINNER of our $1,000,000 lottery! "
        "Click here NOW to claim your prize. Limited time offer. Act FAST! "
        "Call 1-800-FAKE-NUM or reply with your bank details immediately.",

    "🚨 Spam: Urgent Account Alert":
        "URGENT: Your account has been compromised. Verify your identity immediately "
        "by clicking this link and entering your password and social security number. "
        "Failure to do so will result in permanent account suspension within 24 hours.",

    "✅ Ham: Meeting Request":
        "Hi Sarah, hope you're doing well. I wanted to schedule a quick 30-minute call "
        "to discuss the Q3 roadmap and the upcoming feature release. "
        "Would Thursday at 2 PM or Friday morning work for you? Let me know!",

    "✅ Ham: Project Update":
        "Team, just a quick update on the backend refactoring. "
        "We've completed the database migration and the new API endpoints are live in staging. "
        "Please run your integration tests and flag any issues before Friday's release.",
}

# ── Text Preprocessing ────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    if NLTK_AVAILABLE:
        tokens = [_STEMMER.stem(t) for t in tokens if t not in _STOP_WORDS and len(t) > 1]
    else:
        tokens = [t for t in tokens if len(t) > 1]
    return " ".join(tokens)

# ── Model Loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    loaded  = {}
    missing = []

    # TF-IDF vectorizer
    if os.path.exists(MODEL_FILES["tfidf"]):
        with open(MODEL_FILES["tfidf"], "rb") as f:
            loaded["tfidf"] = pickle.load(f)
    else:
        missing.append("tfidf_vectorizer.pkl")

    # Naive Bayes
    if os.path.exists(MODEL_FILES["Naive Bayes"]):
        with open(MODEL_FILES["Naive Bayes"], "rb") as f:
            loaded["Naive Bayes"] = pickle.load(f)
    else:
        missing.append("naive_bayes_model.pkl")

    # SVM
    if os.path.exists(MODEL_FILES["SVM"]):
        with open(MODEL_FILES["SVM"], "rb") as f:
            loaded["SVM"] = pickle.load(f)
    else:
        missing.append("svm_model.pkl")

    # LSTM — only attempt if TensorFlow is installed
    if TF_AVAILABLE:
        if os.path.exists(MODEL_FILES["LSTM"]):
            try:
                loaded["LSTM"] = tf.keras.models.load_model(MODEL_FILES["LSTM"])
            except Exception:
                missing.append("lstm_model.h5 (load error)")
        else:
            missing.append("lstm_model.h5")

        if os.path.exists(MODEL_FILES["tokenizer"]):
            with open(MODEL_FILES["tokenizer"], "rb") as f:
                loaded["lstm_tokenizer"] = pickle.load(f)
        else:
            missing.append("lstm_tokenizer.pkl")

        if os.path.exists(MODEL_FILES["lstm_config"]):
            with open(MODEL_FILES["lstm_config"], "rb") as f:
                loaded["lstm_config"] = pickle.load(f)
        else:
            missing.append("lstm_config.pkl")

    return loaded, missing

# ── Inference ─────────────────────────────────────────────────────────────────

def predict_tfidf(model, tfidf, clean: str):
    vec   = tfidf.transform([clean])
    label = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    return label, float(proba[1])

def predict_lstm(model, tokenizer, config, clean: str):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    max_len = config.get("max_len", 200)
    seq     = tokenizer.texts_to_sequences([clean])
    padded  = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    prob    = float(model.predict(padded, verbose=0)[0][0])
    return (1 if prob >= 0.5 else 0), prob

def run_predictions(text: str, loaded: dict, selected_models: list) -> list:
    clean   = clean_text(text)
    results = []
    for name in selected_models:
        try:
            if name in ("Naive Bayes", "SVM") and name in loaded and "tfidf" in loaded:
                label, prob = predict_tfidf(loaded[name], loaded["tfidf"], clean)
                results.append((name, label, prob))
            elif name == "LSTM" and "LSTM" in loaded:
                label, prob = predict_lstm(
                    loaded["LSTM"], loaded["lstm_tokenizer"], loaded["lstm_config"], clean
                )
                results.append((name, label, prob))
        except Exception as exc:
            results.append((name, -1, str(exc)))
    return results

# ── UI Helpers ────────────────────────────────────────────────────────────────

def spam_badge(label: int, prob: float) -> str:
    if label == 1:
        return f"🚨 **SPAM** ({prob * 100:.1f}% confidence)"
    return f"✅ **HAM** ({(1 - prob) * 100:.1f}% confidence)"

def probability_bar(prob: float):
    color = "#e74c3c" if prob >= 0.5 else "#2ecc71"
    st.markdown(
        f"""
        <div style="background:#e0e0e0;border-radius:8px;height:18px;margin-bottom:4px;">
          <div style="width:{prob*100:.1f}%;background:{color};height:18px;border-radius:8px;
                      transition:width 0.4s ease;"></div>
        </div>
        <p style="margin:0 0 12px 0;font-size:13px;color:#555;">
          Spam probability: <strong>{prob*100:.1f}%</strong>
        </p>
        """,
        unsafe_allow_html=True,
    )

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .hero-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            color: white;
        }
        .hero-header h1 { font-size: 2.2rem; margin: 0 0 0.4rem; }
        .hero-header p  { font-size: 1rem; color: #a0b4cc; margin: 0; }
        .result-card {
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            border: 1px solid #e0e0e0;
            background: #fafafa;
        }
        .result-card.spam { border-left: 5px solid #e74c3c; }
        .result-card.ham  { border-left: 5px solid #2ecc71; }
        section[data-testid="stSidebar"] { background: #f7f9fc; }
        textarea { font-size: 15px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-header">
        <h1>📧 Email Spam Classifier</h1>
        <p>Project #3 · NLP & Machine Learning · Naive Bayes · SVM · LSTM</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load Models ───────────────────────────────────────────────────────────────
with st.spinner("Loading models …"):
    loaded, missing = load_models()

# Only warn about non-LSTM missing files when TF is unavailable
# (LSTM files being absent is expected on Python 3.14)
if missing:
    warn_files = missing if TF_AVAILABLE else [m for m in missing if "lstm" not in m.lower()]
    if warn_files:
        st.warning("⚠️ Some model files were not found: " + ", ".join(f"`{m}`" for m in warn_files))

available_models = [m for m in ("Naive Bayes", "SVM", "LSTM") if m in loaded]

if not available_models:
    st.error(
        f"❌ No models loaded. Make sure `naive_bayes_model.pkl`, `svm_model.pkl`, "
        f"and `tfidf_vectorizer.pkl` are in: `{MODEL_DIR}`"
    )
    # Debug: show what files ARE present to help diagnose
    try:
        files_found = os.listdir(MODEL_DIR)
        st.info("📂 Files found in `" + MODEL_DIR + "`: " + (", ".join(files_found) if files_found else "none"))
    except Exception:
        st.info(f"📂 Could not list directory: `{MODEL_DIR}`")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    selected_models = st.multiselect(
        "Models to use",
        options=available_models,
        default=available_models,
        help="Select one or more classifiers to compare.",
    )

    st.markdown("---")
    st.markdown("### 📋 Try an Example")
    example_choice = st.selectbox("Load example email:", ["— Select —"] + list(EXAMPLE_EMAILS.keys()))

    st.markdown("---")
    st.markdown("### ℹ️ About the Models")
    rows = "| **Naive Bayes** | TF-IDF | ComplementNB |\n| **SVM** | TF-IDF | LinearSVC |"
    if TF_AVAILABLE:
        rows += "\n| **LSTM** | Sequences | Bidirectional LSTM |"
    st.markdown(f"| Model | Input | Notes |\n|---|---|---|\n{rows}")

    if not TF_AVAILABLE:
        st.info("ℹ️ LSTM disabled — TensorFlow requires Python ≤ 3.12. NB & SVM are active.")
    if not NLTK_AVAILABLE:
        st.info("💡 NLTK unavailable — basic tokenisation used.")

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_classify, tab_about = st.tabs(["🔍 Classify", "📖 About"])

# ── Tab 1: Classify ───────────────────────────────────────────────────────────
with tab_classify:
    default_text = ""
    if example_choice != "— Select —":
        default_text = EXAMPLE_EMAILS[example_choice]

    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### ✍️ Enter Email Text")
        email_text = st.text_area(
            label="Email body",
            value=default_text,
            height=280,
            placeholder="Paste or type the email content here …",
            label_visibility="collapsed",
        )

        char_count = len(email_text)
        word_count = len(email_text.split()) if email_text.strip() else 0
        st.caption(f"Characters: **{char_count}** · Words: **{word_count}**")

        st.markdown("&nbsp;")
        classify_btn = st.button("🚀 Classify Email", type="primary", use_container_width=True)

        if email_text.strip():
            with st.expander("🔬 View preprocessed text"):
                st.code(clean_text(email_text), language=None)

    with col_results:
        st.markdown("### 📊 Results")

        if classify_btn:
            if not email_text.strip():
                st.warning("Please enter some email text first.")
            elif not selected_models:
                st.warning("Please select at least one model from the sidebar.")
            else:
                with st.spinner("Running inference …"):
                    predictions = run_predictions(email_text, loaded, selected_models)

                if not predictions:
                    st.error("No predictions could be made. Check model files.")
                else:
                    valid_preds = [(n, l, p) for n, l, p in predictions if isinstance(p, float)]

                    if valid_preds:
                        spam_votes     = sum(1 for _, l, _ in valid_preds if l == 1)
                        avg_prob       = np.mean([p for _, _, p in valid_preds])
                        ensemble_label = 1 if spam_votes > len(valid_preds) / 2 else 0
                        verdict_color  = "#e74c3c" if ensemble_label == 1 else "#2ecc71"
                        verdict_text   = "SPAM" if ensemble_label == 1 else "HAM"
                        verdict_icon   = "🚨" if ensemble_label == 1 else "✅"

                        st.markdown(
                            f"""
                            <div style="background:{verdict_color}22;border:2px solid {verdict_color};
                                        border-radius:14px;padding:1rem 1.4rem;margin-bottom:1.2rem;">
                                <h3 style="margin:0;color:{verdict_color};">
                                    {verdict_icon} Ensemble Verdict: {verdict_text}
                                </h3>
                                <p style="margin:4px 0 0;color:#555;">
                                    {spam_votes}/{len(valid_preds)} models voted Spam ·
                                    Avg spam probability: <strong>{avg_prob*100:.1f}%</strong>
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    for model_name, label, prob in predictions:
                        if isinstance(prob, float):
                            card_cls = "spam" if label == 1 else "ham"
                            st.markdown(f'<div class="result-card {card_cls}">', unsafe_allow_html=True)
                            st.markdown(f"**{model_name}** — {spam_badge(label, prob)}")
                            probability_bar(prob)
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.error(f"**{model_name}** — Error: {prob}")
        else:
            st.info("👈 Enter an email and click **Classify Email** to see predictions.")

# ── Tab 2: About ──────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
        ## 📖 About This Project

        **Project #3 — Email Spam Classification Using NLP and Machine Learning**

        ---

        ### 🔤 Preprocessing Pipeline
        1. **Lowercasing** — normalise case
        2. **URL & email removal** — strip noise
        3. **Number removal** — remove digits
        4. **Punctuation stripping**
        5. **Stop-word removal** *(NLTK)*
        6. **Porter Stemming** *(NLTK)* — reduce to root form

        ---

        ### 🤖 Models

        #### 1. Naive Bayes (ComplementNB)
        Fast, interpretable baseline. Uses TF-IDF word + bigram features with calibrated probabilities.

        #### 2. SVM (LinearSVC)
        High accuracy on sparse text. Calibrated with sigmoid regression for probability output.

        #### 3. LSTM (Bidirectional)
        Captures sequential context and word order. Requires TensorFlow (Python ≤ 3.12).
        *Disabled on Streamlit Cloud (Python 3.14).*
        """
    )

    st.markdown("---")
    st.markdown("### 🩺 System Status")

    status_rows = [
        ("NLTK",              "✅ Available" if NLTK_AVAILABLE else "⚠️ Not installed — basic tokenisation used"),
        ("TensorFlow / LSTM", "✅ Available" if TF_AVAILABLE  else "⚠️ Not available on Python 3.14 — LSTM disabled"),
        ("TF-IDF Vectorizer", "✅ Loaded"    if "tfidf"        in loaded else "❌ Missing"),
        ("Naive Bayes Model", "✅ Loaded"    if "Naive Bayes"  in loaded else "❌ Missing"),
        ("SVM Model",         "✅ Loaded"    if "SVM"          in loaded else "❌ Missing"),
        ("LSTM Model",        "✅ Loaded"    if "LSTM"         in loaded else ("⚠️ Disabled (no TF)" if not TF_AVAILABLE else "❌ Missing")),
        ("Model directory",   f"`{MODEL_DIR}`"),
    ]

    for name, status in status_rows:
        c1, c2 = st.columns([2, 3])
        c1.markdown(f"**{name}**")
        c2.markdown(status)
