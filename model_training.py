"""
=============================================================================
  EMAIL SPAM CLASSIFICATION — MODEL TRAINING
  Project #3 : Email Spam Classification Using NLP and Machine Learning

  Trains three models on preprocessed Enron email data:
    1. Naive Bayes  (ComplementNB + Isotonic Calibration)
    2. SVM          (LinearSVC   + Sigmoid  Calibration)
    3. LSTM         (Bidirectional, 64-dim embeddings)

  Prerequisites (run in order before this script):
    → Preprocess.ipynb       produces  preprocessed_output/
    → feature_extraction.py  produces  feature_extraction_output/

  Outputs → trained_models/
    naive_bayes_model.pkl     Naive Bayes classifier
    svm_model.pkl             SVM classifier
    lstm_model.h5             LSTM Keras model
    lstm_tokenizer.pkl        Keras Tokenizer (word → int mapping)
    lstm_config.pkl           LSTM hyperparams (max_len, vocab_size, …)
    lstm_training_history.pkl Epoch-by-epoch loss / accuracy
    lstm_best_weights.h5      Best checkpoint saved by ModelCheckpoint
    tfidf_vectorizer.pkl      Shared TF-IDF vectorizer (for NB & SVM)

  Usage:
    python model_training.py

  Install dependencies:
    pip install scikit-learn numpy pandas scipy tensorflow
=============================================================================
"""

import os
import pickle
import shutil
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ── Scikit-learn ──────────────────────────────────────────────────────────
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# ── TensorFlow / Keras ────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    Bidirectional, SpatialDropout1D,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")


# =============================================================================
#  CONFIGURATION  — edit paths here if your files live elsewhere
# =============================================================================

PREPROCESS_DIR   = "preprocessed_output"       # output of Preprocess.ipynb
FEATURE_DIR      = "feature_extraction_output" # output of feature_extraction.py
MODEL_OUTPUT_DIR = "trained_models"            # destination for saved models

# ── LSTM hyper-parameters ─────────────────────────────────────────────────
VOCAB_SIZE  = 20_000   # max unique words kept by Keras Tokenizer
MAX_LEN     = 200      # all sequences padded / truncated to this length
EMBED_DIM   = 64       # word-embedding dimension
BATCH_SIZE  = 64
EPOCHS      = 10       # EarlyStopping will stop earlier if val_loss plateaus


# =============================================================================
#  HELPERS
# =============================================================================

def _find(candidates: list[str]) -> str:
    """Return the first existing path from *candidates*.

    Raises FileNotFoundError if none exist.
    """
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find any of: {candidates}\n"
        "Make sure Preprocess.ipynb and feature_extraction.py have been run first."
    )


def _banner(title: str) -> None:
    """Print a section banner to stdout."""
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# =============================================================================
#  STEP 1 — LOAD DATA & FEATURES
# =============================================================================

def load_data():
    """Load all preprocessed artefacts needed for training.

    Returns
    -------
    df_train       : pd.DataFrame  — cleaned training rows (includes 'clean_text')
    df_test        : pd.DataFrame  — cleaned test rows
    X_train_tfidf  : sparse matrix — TF-IDF feature matrix for train
    X_test_tfidf   : sparse matrix — TF-IDF feature matrix for test
    y_train        : np.ndarray    — integer labels (0=ham, 1=spam)
    y_test         : np.ndarray
    tfidf_vectorizer : fitted TfidfVectorizer
    """
    _banner("STEP 1 — LOADING DATA & FEATURES")

    print("  Loading cleaned text DataFrames …")
    df_train = pd.read_csv(_find([
        f"{PREPROCESS_DIR}/train_cleaned.csv", "train_cleaned.csv",
    ]))
    df_test = pd.read_csv(_find([
        f"{PREPROCESS_DIR}/test_cleaned.csv", "test_cleaned.csv",
    ]))

    print("  Loading label arrays …")
    y_train = np.load(_find([
        f"{FEATURE_DIR}/y_train.npy", f"{PREPROCESS_DIR}/y_train.npy",
    ]))
    y_test = np.load(_find([
        f"{FEATURE_DIR}/y_test.npy", f"{PREPROCESS_DIR}/y_test.npy",
    ]))

    print("  Loading TF-IDF feature matrices …")
    X_train_tfidf = sp.load_npz(_find([f"{FEATURE_DIR}/X_train_tfidf.npz"]))
    X_test_tfidf  = sp.load_npz(_find([f"{FEATURE_DIR}/X_test_tfidf.npz"]))

    print("  Loading TF-IDF vectorizer …")
    with open(_find([f"{FEATURE_DIR}/tfidf_vectorizer.pkl"]), "rb") as fh:
        tfidf_vectorizer = pickle.load(fh)

    print(f"\n  Train rows          : {len(df_train)}")
    print(f"  Test  rows          : {len(df_test)}")
    print(f"  X_train_tfidf shape : {X_train_tfidf.shape}")
    print(f"  X_test_tfidf  shape : {X_test_tfidf.shape}")
    print(f"  y_train — ham: {(y_train == 0).sum()}  spam: {(y_train == 1).sum()}")
    print(f"  y_test  — ham: {(y_test  == 0).sum()}  spam: {(y_test  == 1).sum()}")

    return df_train, df_test, X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer


# =============================================================================
#  STEP 2 — NAIVE BAYES
# =============================================================================

def train_naive_bayes(X_train, y_train):
    """Fit a calibrated ComplementNB classifier.

    ComplementNB handles class imbalance better than standard MultinomialNB.
    CalibratedClassifierCV wraps it so predict_proba() returns reliable
    probability estimates (needed for the Streamlit app).

    Parameters
    ----------
    X_train : sparse matrix  TF-IDF features (non-negative values required)
    y_train : np.ndarray     binary labels

    Returns
    -------
    nb_model : fitted CalibratedClassifierCV
    """
    _banner("STEP 2 — TRAINING NAIVE BAYES  (ComplementNB)")

    print("  Input shape  :", X_train.shape, " — sparse TF-IDF matrix")
    print("  Feature values are non-negative ✓  (required for Naive Bayes)")

    nb_base  = ComplementNB(alpha=0.1)          # alpha = Laplace smoothing
    nb_model = CalibratedClassifierCV(
        nb_base, cv=3, method="isotonic"        # isotonic regression calibration
    )

    print("\n  Fitting model on training data …")
    nb_model.fit(X_train, y_train)

    print("  Training complete!")
    print(f"  Model  : {type(nb_model).__name__} wrapping {type(nb_base).__name__}")
    print(f"  alpha  : 0.1  (Laplace smoothing)")
    print(f"  CV     : 3 folds  (isotonic calibration)")

    return nb_model


def save_naive_bayes(nb_model, output_dir: str) -> str:
    """Pickle the trained Naive Bayes model to *output_dir*."""
    path = os.path.join(output_dir, "naive_bayes_model.pkl")
    with open(path, "wb") as fh:
        pickle.dump(nb_model, fh)
    print(f"\n  Naive Bayes model saved → {path}")
    print(f"  File size : {os.path.getsize(path) / 1024:.1f} KB")
    return path


# =============================================================================
#  STEP 3 — SVM
# =============================================================================

def train_svm(X_train, y_train):
    """Fit a calibrated LinearSVC classifier.

    LinearSVC is fast on high-dimensional sparse TF-IDF features.
    class_weight='balanced' compensates for any class imbalance.
    CalibratedClassifierCV adds predict_proba() via sigmoid calibration.

    Parameters
    ----------
    X_train : sparse matrix  L2-normalised TF-IDF features
    y_train : np.ndarray     binary labels

    Returns
    -------
    svm_model : fitted CalibratedClassifierCV
    """
    _banner("STEP 3 — TRAINING SVM  (LinearSVC)")

    print("  Input shape  :", X_train.shape, " — L2-normalised TF-IDF matrix")
    print("  TF-IDF is already L2-normalised ✓  (good for SVM)")

    svm_base  = LinearSVC(
        C=1.0,                  # regularisation strength (higher = less regularised)
        class_weight="balanced",# adjusts weights inversely proportional to class frequencies
        max_iter=2000,          # ensure convergence on large datasets
        random_state=42,
    )
    svm_model = CalibratedClassifierCV(
        svm_base, cv=3, method="sigmoid"   # Platt scaling
    )

    print("\n  Fitting model on training data …")
    print("  (This may take a minute on large datasets)")
    svm_model.fit(X_train, y_train)

    print("  Training complete!")
    print(f"  Model        : {type(svm_model).__name__} wrapping {type(svm_base).__name__}")
    print(f"  C            : 1.0")
    print(f"  class_weight : balanced")
    print(f"  CV           : 3 folds  (sigmoid calibration)")

    return svm_model


def save_svm(svm_model, output_dir: str) -> str:
    """Pickle the trained SVM model to *output_dir*."""
    path = os.path.join(output_dir, "svm_model.pkl")
    with open(path, "wb") as fh:
        pickle.dump(svm_model, fh)
    print(f"\n  SVM model saved → {path}")
    print(f"  File size : {os.path.getsize(path) / 1024:.1f} KB")
    return path


# =============================================================================
#  STEP 4 — LSTM
# =============================================================================

def tokenize_for_lstm(train_texts, test_texts):
    """Build a Keras Tokenizer on training data and pad both splits.

    Steps
    -----
    1. Fit Tokenizer on training corpus only (avoids data leakage).
    2. Convert texts to integer sequences (unknown words → <OOV> token).
    3. Pad / truncate all sequences to MAX_LEN.

    Returns
    -------
    tokenizer        : fitted Keras Tokenizer
    vocab_size_actual: int  — capped at VOCAB_SIZE
    X_train_padded   : np.ndarray  shape (n_train, MAX_LEN)
    X_test_padded    : np.ndarray  shape (n_test,  MAX_LEN)
    """
    _banner("STEP 4a — LSTM TOKENIZATION")

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    vocab_size_actual = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)

    print(f"  Vocabulary built from training corpus")
    print(f"  Total unique words  : {len(tokenizer.word_index):,}")
    print(f"  Vocabulary size used: {vocab_size_actual:,}  (capped at {VOCAB_SIZE:,})")

    print(f"\n  Converting texts to integer sequences …")
    X_train_seq = tokenizer.texts_to_sequences(train_texts)
    X_test_seq  = tokenizer.texts_to_sequences(test_texts)

    print(f"  Padding / truncating to length {MAX_LEN} …")
    X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_padded  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding="post", truncating="post")

    print(f"\n  X_train_padded : {X_train_padded.shape}  (samples × sequence_length)")
    print(f"  X_test_padded  : {X_test_padded.shape}")

    return tokenizer, vocab_size_actual, X_train_padded, X_test_padded


def build_lstm_model(vocab_size_actual: int) -> tf.keras.Model:
    """Construct and compile the Bidirectional LSTM model.

    Architecture
    ------------
    Input (MAX_LEN tokens)
      → Embedding        (vocab_size × EMBED_DIM)
      → SpatialDropout1D (0.2)
      → Bidirectional LSTM (64 units, dropout=0.2)
      → Dense            (128, ReLU)
      → Dropout          (0.3)
      → Dense            (1, sigmoid)  →  spam probability

    Returns
    -------
    lstm_model : compiled tf.keras.Model
    """
    _banner("STEP 4b — BUILD LSTM ARCHITECTURE")

    tf.random.set_seed(42)
    np.random.seed(42)

    model = Sequential([
        # Converts integer token IDs to dense EMBED_DIM-dimensional vectors
        Embedding(
            input_dim=vocab_size_actual,
            output_dim=EMBED_DIM,
            input_length=MAX_LEN,
            name="embedding",
        ),
        # Drops entire embedding vectors (more effective than per-element dropout for text)
        SpatialDropout1D(0.2, name="spatial_dropout"),
        # Reads the sequence both left→right and right→left
        Bidirectional(
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            name="bilstm",
        ),
        Dense(128, activation="relu", name="dense_1"),
        Dropout(0.3, name="dropout"),
        # Single sigmoid output: > 0.5 → spam, < 0.5 → ham
        Dense(1, activation="sigmoid", name="output"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def get_lstm_callbacks(output_dir: str) -> list:
    """Create training callbacks.

    EarlyStopping      — halts training when val_loss stops improving (patience=3)
    ModelCheckpoint    — saves the epoch with the lowest val_loss
    ReduceLROnPlateau  — halves the learning rate when val_loss stagnates (patience=2)
    """
    checkpoint_path = os.path.join(output_dir, "lstm_best_weights.h5")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\n  Callbacks configured:")
    print(f"  EarlyStopping     — patience=3,  monitors val_loss")
    print(f"  ModelCheckpoint   — best weights → {checkpoint_path}")
    print(f"  ReduceLROnPlateau — factor=0.5,  patience=2")

    return callbacks


def train_lstm(model, X_train_padded, y_train, callbacks):
    """Run the LSTM training loop.

    Parameters
    ----------
    model           : compiled Keras model
    X_train_padded  : np.ndarray  padded token sequences
    y_train         : np.ndarray  binary labels
    callbacks       : list        Keras callbacks

    Returns
    -------
    history : Keras History object
    """
    _banner("STEP 4c — TRAINING LSTM")

    print(f"  Training samples : {len(X_train_padded)}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Max epochs       : {EPOCHS}  (EarlyStopping may end sooner)")
    print(f"  Validation split : 20% of training data\n")

    history = model.fit(
        X_train_padded,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    epochs_done   = len(history.history["loss"])
    best_val_loss = min(history.history["val_loss"])
    best_val_acc  = max(history.history["val_accuracy"])

    print(f"\n  Training complete!")
    print(f"  Epochs trained    : {epochs_done}")
    print(f"  Best val loss     : {best_val_loss:.4f}")
    print(f"  Best val accuracy : {best_val_acc:.4f}")

    return history


def save_lstm(model, tokenizer, vocab_size_actual, history, output_dir: str):
    """Persist the LSTM model, tokenizer, config, and training history."""
    _banner("STEP 4d — SAVING LSTM ARTEFACTS")

    # Full Keras model (architecture + weights)
    model_path = os.path.join(output_dir, "lstm_model.h5")
    model.save(model_path)
    print(f"  LSTM model saved          → {model_path}")

    # Tokenizer  (word → integer mapping, required in Streamlit app)
    tokenizer_path = os.path.join(output_dir, "lstm_tokenizer.pkl")
    with open(tokenizer_path, "wb") as fh:
        pickle.dump(tokenizer, fh)
    print(f"  LSTM tokenizer saved      → {tokenizer_path}")

    # Config dict  (needed to reconstruct pad_sequences in Streamlit)
    config = {
        "vocab_size": vocab_size_actual,
        "max_len"   : MAX_LEN,
        "embed_dim" : EMBED_DIM,
    }
    config_path = os.path.join(output_dir, "lstm_config.pkl")
    with open(config_path, "wb") as fh:
        pickle.dump(config, fh)
    print(f"  LSTM config saved         → {config_path}")

    # Training history  (for evaluation / plotting notebook)
    history_path = os.path.join(output_dir, "lstm_training_history.pkl")
    with open(history_path, "wb") as fh:
        pickle.dump(history.history, fh)
    print(f"  Training history saved    → {history_path}")


# =============================================================================
#  STEP 5 — SAVE SHARED ARTEFACTS
# =============================================================================

def save_shared_artefacts(tfidf_vectorizer, output_dir: str):
    """Copy the shared TF-IDF vectorizer into trained_models/.

    Both Naive Bayes and SVM require this to transform raw text at
    inference time in the Streamlit app.
    """
    _banner("STEP 5 — SAVING SHARED ARTEFACTS")

    dest = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    with open(dest, "wb") as fh:
        pickle.dump(tfidf_vectorizer, fh)
    print(f"  TF-IDF vectorizer saved → {dest}")
    print("  (Shared by Naive Bayes and SVM)")


# =============================================================================
#  STEP 6 — SUMMARY
# =============================================================================

def print_summary(output_dir: str):
    """Print a table of all saved files with sizes."""
    _banner("STEP 6 — TRAINING SUMMARY")

    saved_files = [
        ("naive_bayes_model.pkl",     "Naive Bayes  (ComplementNB + Calibration)"),
        ("svm_model.pkl",             "SVM          (LinearSVC   + Calibration)"),
        ("lstm_model.h5",             "LSTM         (Bidirectional, 64-dim)"),
        ("lstm_tokenizer.pkl",        "LSTM Tokenizer  (word → int mapping)"),
        ("lstm_config.pkl",           "LSTM Config     (vocab_size, max_len, …)"),
        ("lstm_training_history.pkl", "LSTM Training History"),
        ("lstm_best_weights.h5",      "LSTM Best Checkpoint"),
        ("tfidf_vectorizer.pkl",      "Shared TF-IDF Vectorizer  (NB + SVM)"),
    ]

    print(f"\n  Output directory : {os.path.abspath(output_dir)}/\n")
    for fname, desc in saved_files:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  ✓  {fname:<40}  {size_kb:>8.1f} KB  — {desc}")
        else:
            print(f"  ✗  {fname:<40}  NOT FOUND")

    print("""
  ── How to load models in the Streamlit app ───────────────────────

  import pickle, tensorflow as tf
  from tensorflow.keras.preprocessing.sequence import pad_sequences

  # Naive Bayes
  nb    = pickle.load(open("trained_models/naive_bayes_model.pkl", "rb"))

  # SVM
  svm   = pickle.load(open("trained_models/svm_model.pkl", "rb"))

  # LSTM
  lstm  = tf.keras.models.load_model("trained_models/lstm_model.h5")
  tok   = pickle.load(open("trained_models/lstm_tokenizer.pkl", "rb"))
  cfg   = pickle.load(open("trained_models/lstm_config.pkl",    "rb"))

  # TF-IDF vectorizer  (for Naive Bayes & SVM)
  tfidf = pickle.load(open("trained_models/tfidf_vectorizer.pkl", "rb"))
""")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  EMAIL SPAM CLASSIFICATION — MODEL TRAINING")
    print("  Project #3")
    print("=" * 65)
    print(f"  TensorFlow version : {tf.__version__}")
    print(f"  NumPy version      : {np.__version__}")

    # ── Create output directory ───────────────────────────────────────────
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    print(f"\n  Models will be saved to: {os.path.abspath(MODEL_OUTPUT_DIR)}")

    # ── 1. Load data ──────────────────────────────────────────────────────
    (
        df_train, df_test,
        X_train_tfidf, X_test_tfidf,
        y_train, y_test,
        tfidf_vectorizer,
    ) = load_data()

    train_texts = df_train["clean_text"].fillna("").values
    test_texts  = df_test["clean_text"].fillna("").values

    # ── 2. Naive Bayes ────────────────────────────────────────────────────
    nb_model = train_naive_bayes(X_train_tfidf, y_train)
    save_naive_bayes(nb_model, MODEL_OUTPUT_DIR)

    # ── 3. SVM ────────────────────────────────────────────────────────────
    svm_model = train_svm(X_train_tfidf, y_train)
    save_svm(svm_model, MODEL_OUTPUT_DIR)

    # ── 4. LSTM ───────────────────────────────────────────────────────────
    tokenizer, vocab_size_actual, X_train_padded, X_test_padded = tokenize_for_lstm(
        train_texts, test_texts
    )
    lstm_model = build_lstm_model(vocab_size_actual)
    callbacks  = get_lstm_callbacks(MODEL_OUTPUT_DIR)
    history    = train_lstm(lstm_model, X_train_padded, y_train, callbacks)
    save_lstm(lstm_model, tokenizer, vocab_size_actual, history, MODEL_OUTPUT_DIR)

    # ── 5. Shared artefacts ───────────────────────────────────────────────
    save_shared_artefacts(tfidf_vectorizer, MODEL_OUTPUT_DIR)

    # ── 6. Summary ────────────────────────────────────────────────────────
    print_summary(MODEL_OUTPUT_DIR)

    print("\n  All models trained and saved successfully ✓\n")


if __name__ == "__main__":
    main()
