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

# ── Safe Pickle Loader ────────────────────────────────────────────────────────

def safe_pickle_load(filepath: str):
    """
    Load a pickle file safely.
    Returns (object, error_message).
    error_message is None on success, a string on failure.
    """
    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj, None
    except ModuleNotFoundError as e:
        return None, f"Missing module: {e} — retrain with matching library versions"
    except pickle.UnpicklingError as e:
        return None, f"Corrupt pickle file: {e}"
    except AttributeError as e:
        return None, f"Version mismatch: {e} — retrain models on matching scikit-learn version"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# ── Model Loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    loaded  = {}
    missing = []
    errors  = {}

    # TF-IDF vectorizer
    if os.path.exists(MODEL_FILES["tfidf"]):
        obj, err = safe_pickle_load(MODEL_FILES["tfidf"])
        if obj is not None:
            loaded["tfidf"] = obj
        else:
            missing.append("tfidf_vectorizer.pkl")
            errors["tfidf_vectorizer.pkl"] = err
    else:
        missing.append("tfidf_vectorizer.pkl")

    # Naive Bayes
    if os.path.exists(MODEL_FILES["Naive Bayes"]):
        obj, err = safe_pickle_load(MODEL_FILES["Naive Bayes"])
        if obj is not None:
            loaded["Naive Bayes"] = obj
        else:
            missing.append("naive_bayes_model.pkl")
            errors["naive_bayes_model.pkl"] = err
    else:
        missing.append("naive_bayes_model.pkl")

    # SVM
    if os.path.exists(MODEL_FILES["SVM"]):
        obj, err = safe_pickle_load(MODEL_FILES["SVM"])
        if obj is not None:
            loaded["SVM"] = obj
        else:
            missing.append("svm_model.pkl")
            errors["svm_model.pkl"] = err
    else:
        missing.append("svm_model.pkl")

    # LSTM — only attempt if TensorFlow is installed
    if TF_AVAILABLE:
        if os.path.exists(MODEL_FILES["LSTM"]):
            try:
                loaded["LSTM"] = tf.keras.models.load_model(MODEL_FILES["LSTM"])
            except Exception as e:
                missing.append("lstm_model.h5")
                errors["lstm_model.h5"] = str(e)
        else:
            missing.append("lstm_model.h5")

        if os.path.exists(MODEL_FILES["tokenizer"]):
            obj, err = safe_pickle_load(MODEL_FILES["tokenizer"])
            if obj is not None:
                loaded["lstm_tokenizer"] = obj
            else:
                missing.append("lstm_tokenizer.pkl")
                errors["lstm_tokenizer.pkl"] = err
        else:
            missing.append("lstm_tokenizer.pkl")

        if os.path.exists(MODEL_FILES["lstm_config"]):
            obj, err = safe_pickle_load(MODEL_FILES["lstm_config"])
            if obj is not None:
                loaded["lstm_config"] = obj
            else:
                missing.append("lstm_config.pkl")
                errors["lstm_config.pkl"] = err
        else:
            missing.append("lstm_config.pkl")

    return loaded, missing, errors

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
            padding: 2rem 2.5rem; border-radius: 16px;
            margin-bottom: 2rem; color: white;
        }
        .hero-header h1 { font-size: 2.2rem; margin: 0 0 0.4rem; }
        .hero-header p  { font-size: 1rem; color: #a0b4cc; margin: 0; }
        .result-card {
            border-radius: 12px; padding: 1.2rem 1.4rem;
            margin-bottom: 1rem; border: 1px solid #e0e0e0; background: #fafafa;
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
    loaded, missing, load_errors = load_models()

# Show errors with specific reasons so the user knows what to fix
if load_errors:
    with st.expander("⚠️ Model loading errors (click to expand)", expanded=True):
        for fname, reason in load_errors.items():
            st.error(f"**{fname}**: {reason}")
        st.info(
            "💡 **Most likely cause:** The `.pkl` files were saved with a different version of "
            "scikit-learn than what is currently installed.  \n"
            "**Fix:** Re-run your training notebook (`Model_Training.ipynb`) in an environment "
            "matching `requirements.txt` (scikit-learn==1.4.2, Python 3.11), then re-upload the `.pkl` files."
        )

# Warn about missing files (not pkl errors — those are shown above)
non_error_missing = [m for m in missing if m not in load_errors]
if non_error_missing:
    lstm_only = all("lstm" in m.lower() for m in non_error_missing)
    if not (lstm_only and not TF_AVAILABLE):
        st.warning("⚠️ Model files not found: " + ", ".join(f"`{m}`" for m in non_error_missing))

available_models = [m for m in ("Naive Bayes", "SVM", "LSTM") if m in loaded]

if not available_models:
    st.error(f"❌ No models could be loaded. See errors above.")
    st.markdown(f"**Model directory being checked:** `{MODEL_DIR}`")
    try:
        files = os.listdir(MODEL_DIR)
        st.markdown("**Files found:** " + (", ".join(f"`{f}`" for f in sorted(files)) if files else "none"))
    except Exception:
        pass
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
        st.info("ℹ️ LSTM disabled — TensorFlow requires Python ≤ 3.12.")
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
                    st.error("No predictions could be made.")
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
        1. **Lowercasing**
        2. **URL & email removal**
        3. **Number removal**
        4. **Punctuation stripping**
        5. **Stop-word removal** *(NLTK)*
        6. **Porter Stemming** *(NLTK)*

        ---

        ### 🤖 Models

        #### 1. Naive Bayes (ComplementNB)
        Fast, interpretable baseline. TF-IDF features with calibrated probabilities.

        #### 2. SVM (LinearSVC)
        High accuracy on sparse text. Calibrated with sigmoid for probability output.

        #### 3. LSTM (Bidirectional)
        Captures sequential word context. Requires TensorFlow (Python ≤ 3.12).
        """
    )

    st.markdown("---")
    st.markdown("### 🩺 System Status")

    status_rows = [
        ("NLTK",              "✅ Available" if NLTK_AVAILABLE else "⚠️ Not installed"),
        ("TensorFlow / LSTM", "✅ Available" if TF_AVAILABLE  else "⚠️ Disabled (Python 3.14 incompatible)"),
        ("TF-IDF Vectorizer", "✅ Loaded"    if "tfidf"       in loaded else "❌ Failed to load"),
        ("Naive Bayes Model", "✅ Loaded"    if "Naive Bayes" in loaded else "❌ Failed to load"),
        ("SVM Model",         "✅ Loaded"    if "SVM"         in loaded else "❌ Failed to load"),
        ("LSTM Model",        "✅ Loaded"    if "LSTM"        in loaded else ("⚠️ Disabled" if not TF_AVAILABLE else "❌ Failed to load")),
        ("Model directory",   f"`{MODEL_DIR}`"),
    ]

    for name, status in status_rows:
        c1, c2 = st.columns([2, 3])
        c1.markdown(f"**{name}**")
        c2.markdown(status)

    if load_errors:
        st.markdown("---")
        st.markdown("### ❌ Load Errors")
        for fname, reason in load_errors.items():
            st.error(f"**{fname}**: {reason}")
