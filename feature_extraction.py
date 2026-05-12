"""
=============================================================================
  EMAIL SPAM CLASSIFICATION — FEATURE EXTRACTION PIPELINE
  Project #3 : Email Spam Classification Using NLP and Machine Learning

  Input  : Preprocessed artifacts from Preprocess.ipynb
           - train_cleaned.csv / test_cleaned.csv   (cleaned text + labels)
           - X_train_count.npz / X_test_count.npz   (CountVectorizer matrix)
           - count_vectorizer.pkl                    (fitted CountVectorizer)
           - y_train.npy / y_test.npy                (labels)
           - y_train_balanced.npy                    (SMOTE-balanced labels)
           - vocabulary.csv                          (token → index mapping)

  Extractions:
    1. TF-IDF  (unigram + bigram, up to 10 000 features)
    2. TF-IDF  (character n-grams 2-4, up to 5 000 features)
    3. Word Embeddings  — mean-pooled GloVe-style vectors
       (falls back to a trained Word2Vec if GloVe file absent)
    4. Combined feature matrix  (TF-IDF + embeddings)

  Outputs  → feature_extraction_output/
    - X_train_tfidf.npz            TF-IDF word n-gram (sparse)
    - X_test_tfidf.npz
    - X_train_tfidf_char.npz       TF-IDF char n-gram (sparse)
    - X_test_tfidf_char.npz
    - X_train_embeddings.npy       Mean-pooled word embeddings (dense)
    - X_test_embeddings.npy
    - X_train_combined.npz         TF-IDF (word) + embeddings stacked
    - X_test_combined.npz
    - tfidf_vectorizer.pkl         Fitted word TF-IDF vectorizer
    - tfidf_char_vectorizer.pkl    Fitted char TF-IDF vectorizer
    - feature_summary.txt          Human-readable summary

=============================================================================
"""

import os
import re
import json
import pickle
import warnings
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ── Optional: gensim for Word2Vec fallback ────────────────────────────────
try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  — edit paths here if your files live elsewhere
# ─────────────────────────────────────────────────────────────────────────

INPUT_DIR  = "preprocessed_output"   # folder produced by Preprocess.ipynb
OUTPUT_DIR = "feature_extraction_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional: path to a pre-trained GloVe text file (e.g. glove.6B.100d.txt)
# If not found, the script trains a lightweight Word2Vec on the corpus instead.
GLOVE_PATH   = os.environ.get("GLOVE_PATH", "glove.6B.100d.txt")
EMBED_DIM    = 100          # must match GloVe file if using GloVe

# TF-IDF hyper-parameters
TFIDF_WORD_MAX_FEATURES = 10_000
TFIDF_CHAR_MAX_FEATURES = 5_000
TFIDF_WORD_NGRAM        = (1, 2)   # unigrams + bigrams
TFIDF_CHAR_NGRAM        = (2, 4)   # character n-grams 2-4
TFIDF_MIN_DF            = 2
TFIDF_MAX_DF            = 0.95
TFIDF_SUBLINEAR_TF      = True     # log(tf) + 1 smoothing

# Word2Vec hyper-params (fallback when GloVe absent)
W2V_VECTOR_SIZE = EMBED_DIM
W2V_WINDOW      = 5
W2V_MIN_COUNT   = 2
W2V_WORKERS     = 4
W2V_EPOCHS      = 5

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="  %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────

def _find(candidates):
    """Return the first existing path from a list of candidates."""
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find any of: {candidates}\n"
        f"Make sure INPUT_DIR='{INPUT_DIR}' points to your preprocessed_output folder."
    )


def load_preprocessed():
    """Load all artefacts produced by the preprocessing notebook."""
    log.info("Loading preprocessed artefacts ...")

    train_csv = _find([
        f"{INPUT_DIR}/train_cleaned.csv",
        "train_cleaned.csv",
    ])
    test_csv = _find([
        f"{INPUT_DIR}/test_cleaned.csv",
        "test_cleaned.csv",
    ])

    df_train = pd.read_csv(train_csv)
    df_test  = pd.read_csv(test_csv)

    y_train = np.load(_find([f"{INPUT_DIR}/y_train.npy", "y_train.npy"]))
    y_test  = np.load(_find([f"{INPUT_DIR}/y_test.npy",  "y_test.npy"]))

    y_train_balanced = np.load(_find([
        f"{INPUT_DIR}/y_train_balanced.npy", "y_train_balanced.npy"
    ]))

    X_train_count = sp.load_npz(_find([
        f"{INPUT_DIR}/X_train_count.npz", "X_train_count.npz"
    ]))
    X_test_count = sp.load_npz(_find([
        f"{INPUT_DIR}/X_test_count.npz", "X_test_count.npz"
    ]))

    with open(_find([
        f"{INPUT_DIR}/count_vectorizer.pkl", "count_vectorizer.pkl"
    ]), "rb") as fh:
        count_vec = pickle.load(fh)

    log.info(f"Train rows : {len(df_train)}  |  Test rows : {len(df_test)}")
    log.info(f"X_train_count shape : {X_train_count.shape}")
    log.info(f"X_test_count  shape : {X_test_count.shape}")
    log.info(f"y_train dist  — ham: {(y_train==0).sum()}  spam: {(y_train==1).sum()}")
    log.info(f"y_test  dist  — ham: {(y_test==0).sum()}   spam: {(y_test==1).sum()}")

    return (
        df_train, df_test,
        X_train_count, X_test_count,
        y_train, y_test, y_train_balanced,
        count_vec,
    )


# ─────────────────────────────────────────────────────────────────────────
#  FEATURE BLOCK 1 — TF-IDF  (word n-grams)
# ─────────────────────────────────────────────────────────────────────────

def extract_tfidf_word(train_texts, test_texts):
    """
    Fit a TF-IDF vectorizer on training data and transform both splits.

    Features:
      - analyzer  : word
      - n-gram     : (1, 2)  — unigrams + bigrams
      - max_features : 10 000
      - sublinear_tf : True  (log(tf) smoothing)
      - norm         : 'l2'

    Returns sparse matrices X_train, X_test and the fitted vectorizer.
    """
    print("\n" + "=" * 65)
    print("  FEATURE BLOCK 1 — TF-IDF  (word unigrams + bigrams)")
    print("=" * 65)

    tfidf = TfidfVectorizer(
        analyzer      = "word",
        ngram_range   = TFIDF_WORD_NGRAM,
        max_features  = TFIDF_WORD_MAX_FEATURES,
        min_df        = TFIDF_MIN_DF,
        max_df        = TFIDF_MAX_DF,
        sublinear_tf  = TFIDF_SUBLINEAR_TF,
        norm          = "l2",
        token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )

    log.info("Fitting TF-IDF word vectorizer on TRAIN ...")
    X_train = tfidf.fit_transform(train_texts)
    X_test  = tfidf.transform(test_texts)

    vocab_size = len(tfidf.vocabulary_)
    log.info(f"Vocabulary size   : {vocab_size}")
    log.info(f"X_train_tfidf     : {X_train.shape}  (sparse, nnz={X_train.nnz})")
    log.info(f"X_test_tfidf      : {X_test.shape}")

    # Top-10 features by mean TF-IDF score on train
    mean_tfidf = np.asarray(X_train.mean(axis=0)).ravel()
    top_idx    = mean_tfidf.argsort()[::-1][:10]
    inv_vocab  = {v: k for k, v in tfidf.vocabulary_.items()}
    top_tokens = [inv_vocab[i] for i in top_idx]
    log.info(f"Top-10 TF-IDF tokens: {top_tokens}")

    return X_train, X_test, tfidf


# ─────────────────────────────────────────────────────────────────────────
#  FEATURE BLOCK 2 — TF-IDF  (character n-grams)
# ─────────────────────────────────────────────────────────────────────────

def extract_tfidf_char(train_texts, test_texts):
    """
    Character-level TF-IDF captures morphological patterns and is robust
    against deliberate misspellings used in spam (e.g. "v1agra", "fr33").

    Features:
      - analyzer    : char_wb  (char n-grams within word boundaries)
      - n-gram       : (2, 4)
      - max_features : 5 000
    """
    print("\n" + "=" * 65)
    print("  FEATURE BLOCK 2 — TF-IDF  (character n-grams 2-4)")
    print("=" * 65)

    tfidf_char = TfidfVectorizer(
        analyzer     = "char_wb",
        ngram_range  = TFIDF_CHAR_NGRAM,
        max_features = TFIDF_CHAR_MAX_FEATURES,
        min_df       = TFIDF_MIN_DF,
        max_df       = TFIDF_MAX_DF,
        sublinear_tf = TFIDF_SUBLINEAR_TF,
        norm         = "l2",
    )

    log.info("Fitting TF-IDF char vectorizer on TRAIN ...")
    X_train = tfidf_char.fit_transform(train_texts)
    X_test  = tfidf_char.transform(test_texts)

    log.info(f"Char vocab size   : {len(tfidf_char.vocabulary_)}")
    log.info(f"X_train_tfidf_char: {X_train.shape}  (sparse, nnz={X_train.nnz})")
    log.info(f"X_test_tfidf_char : {X_test.shape}")

    return X_train, X_test, tfidf_char


# ─────────────────────────────────────────────────────────────────────────
#  FEATURE BLOCK 3 — Word Embeddings
# ─────────────────────────────────────────────────────────────────────────

def _load_glove(path, embed_dim):
    """Parse a GloVe text file into a {word: np.ndarray} dictionary."""
    log.info(f"Loading GloVe vectors from '{path}' ...")
    embeddings = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            if len(parts) != embed_dim + 1:
                continue
            word = parts[0]
            vec  = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    log.info(f"Loaded {len(embeddings):,} GloVe vectors  (dim={embed_dim})")
    return embeddings


def _train_word2vec(corpus_tokens, embed_dim):
    """Train a Word2Vec model on the tokenized corpus."""
    if not GENSIM_AVAILABLE:
        raise ImportError(
            "gensim is required for Word2Vec training.\n"
            "Install it with:  pip install gensim\n"
            "Or supply a GloVe file at GLOVE_PATH."
        )
    log.info(f"Training Word2Vec  (dim={embed_dim}, window={W2V_WINDOW}, "
             f"epochs={W2V_EPOCHS}) ...")
    model = Word2Vec(
        sentences   = corpus_tokens,
        vector_size = embed_dim,
        window      = W2V_WINDOW,
        min_count   = W2V_MIN_COUNT,
        workers     = W2V_WORKERS,
        epochs      = W2V_EPOCHS,
        seed        = 42,
    )
    log.info(f"Word2Vec vocabulary : {len(model.wv):,} words")
    # Return a simple dict-like interface identical to GloVe lookup
    return {w: model.wv[w] for w in model.wv.index_to_key}


def _mean_pool(texts, embeddings, embed_dim):
    """
    For each document, compute the mean of all token embedding vectors.
    Tokens absent from the embedding space are silently skipped.
    Documents with no known tokens receive a zero vector.
    """
    matrix = np.zeros((len(texts), embed_dim), dtype=np.float32)
    oov_total = 0
    token_total = 0

    for i, text in enumerate(texts):
        tokens = text.split() if isinstance(text, str) else []
        vecs   = []
        for tok in tokens:
            token_total += 1
            if tok in embeddings:
                vecs.append(embeddings[tok])
            else:
                oov_total += 1
        if vecs:
            matrix[i] = np.mean(vecs, axis=0)

    oov_rate = oov_total / token_total * 100 if token_total > 0 else 0
    log.info(f"OOV rate : {oov_rate:.1f}%  ({oov_total:,} / {token_total:,} tokens)")
    return matrix


def extract_embeddings(train_texts, test_texts, embed_dim=EMBED_DIM):
    """
    Build dense mean-pooled embedding matrices for train and test.

    Strategy (in priority order):
      1. Load GloVe from GLOVE_PATH if the file exists.
      2. Otherwise train a lightweight Word2Vec on the training corpus.

    The final vectors are L2-normalised row-wise.

    Returns dense numpy arrays X_train_emb (n_train × embed_dim),
                                X_test_emb  (n_test  × embed_dim).
    """
    print("\n" + "=" * 65)
    print("  FEATURE BLOCK 3 — Word Embeddings  (mean-pooled)")
    print("=" * 65)

    # --- Obtain embedding lookup -----------------------------------------
    if os.path.exists(GLOVE_PATH):
        log.info("GloVe file found — using pre-trained GloVe vectors.")
        embeddings = _load_glove(GLOVE_PATH, embed_dim)
    else:
        log.info(
            f"GloVe file not found at '{GLOVE_PATH}'.\n"
            "  Falling back to training Word2Vec on the training corpus."
        )
        corpus_tokens = [
            t.split() for t in train_texts if isinstance(t, str)
        ]
        embeddings = _train_word2vec(corpus_tokens, embed_dim)

    # --- Mean-pool ----------------------------------------------------------
    log.info("Building train embedding matrix ...")
    X_train_emb = _mean_pool(train_texts.tolist(), embeddings, embed_dim)

    log.info("Building test embedding matrix ...")
    X_test_emb  = _mean_pool(test_texts.tolist(),  embeddings, embed_dim)

    # --- L2 normalise -------------------------------------------------------
    X_train_emb = normalize(X_train_emb, norm="l2")
    X_test_emb  = normalize(X_test_emb,  norm="l2")

    log.info(f"X_train_embeddings : {X_train_emb.shape}  (dense)")
    log.info(f"X_test_embeddings  : {X_test_emb.shape}")

    return X_train_emb, X_test_emb


# ─────────────────────────────────────────────────────────────────────────
#  FEATURE BLOCK 4 — Combined  (TF-IDF word + Embeddings)
# ─────────────────────────────────────────────────────────────────────────

def combine_features(X_tfidf_train, X_tfidf_test,
                     X_emb_train,   X_emb_test):
    """
    Horizontally stack sparse TF-IDF features with dense embedding
    features to produce a single combined feature matrix.

    TF-IDF  (sparse, n × 10 000)
    Embeddings (dense, n × 100)
    → Combined (sparse, n × 10 100)
    """
    print("\n" + "=" * 65)
    print("  FEATURE BLOCK 4 — Combined  (TF-IDF + Embeddings)")
    print("=" * 65)

    # Convert dense embeddings to sparse for horizontal stack
    X_emb_train_sp = sp.csr_matrix(X_emb_train)
    X_emb_test_sp  = sp.csr_matrix(X_emb_test)

    X_train_combined = sp.hstack([X_tfidf_train, X_emb_train_sp], format="csr")
    X_test_combined  = sp.hstack([X_tfidf_test,  X_emb_test_sp],  format="csr")

    log.info(f"X_train_combined : {X_train_combined.shape}")
    log.info(f"X_test_combined  : {X_test_combined.shape}")

    return X_train_combined, X_test_combined


# ─────────────────────────────────────────────────────────────────────────
#  SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────

def save_outputs(
    X_train_tfidf,      X_test_tfidf,       tfidf_vec,
    X_train_tfidf_char, X_test_tfidf_char,  tfidf_char_vec,
    X_train_emb,        X_test_emb,
    X_train_combined,   X_test_combined,
    y_train,            y_test,             y_train_balanced,
):
    print("\n" + "=" * 65)
    print("  SAVING FEATURE EXTRACTION OUTPUTS")
    print("=" * 65)

    out = OUTPUT_DIR

    # Sparse TF-IDF matrices
    sp.save_npz(f"{out}/X_train_tfidf.npz",      X_train_tfidf)
    sp.save_npz(f"{out}/X_test_tfidf.npz",       X_test_tfidf)
    sp.save_npz(f"{out}/X_train_tfidf_char.npz", X_train_tfidf_char)
    sp.save_npz(f"{out}/X_test_tfidf_char.npz",  X_test_tfidf_char)
    sp.save_npz(f"{out}/X_train_combined.npz",   X_train_combined)
    sp.save_npz(f"{out}/X_test_combined.npz",    X_test_combined)
    log.info("Saved sparse matrices  (.npz)")

    # Dense embedding matrices
    np.save(f"{out}/X_train_embeddings.npy", X_train_emb)
    np.save(f"{out}/X_test_embeddings.npy",  X_test_emb)
    log.info("Saved dense embedding matrices  (.npy)")

    # Labels (also copy for convenience)
    np.save(f"{out}/y_train.npy",          y_train)
    np.save(f"{out}/y_test.npy",           y_test)
    np.save(f"{out}/y_train_balanced.npy", y_train_balanced)
    log.info("Saved label arrays  (.npy)")

    # Fitted vectorizers
    with open(f"{out}/tfidf_vectorizer.pkl",      "wb") as fh:
        pickle.dump(tfidf_vec,      fh)
    with open(f"{out}/tfidf_char_vectorizer.pkl", "wb") as fh:
        pickle.dump(tfidf_char_vec, fh)
    log.info("Saved fitted TF-IDF vectorizers  (.pkl)")

    # Human-readable summary
    summary_lines = [
        "=" * 65,
        "  FEATURE EXTRACTION SUMMARY",
        "=" * 65,
        f"  Output directory    : {os.path.abspath(out)}",
        "",
        "  ── TF-IDF Word Features ──",
        f"  Vectorizer          : word, ngram=(1,2), max_features={TFIDF_WORD_MAX_FEATURES}",
        f"  X_train_tfidf       : {X_train_tfidf.shape}",
        f"  X_test_tfidf        : {X_test_tfidf.shape}",
        "",
        "  ── TF-IDF Char Features ──",
        f"  Vectorizer          : char_wb, ngram=(2,4), max_features={TFIDF_CHAR_MAX_FEATURES}",
        f"  X_train_tfidf_char  : {X_train_tfidf_char.shape}",
        f"  X_test_tfidf_char   : {X_test_tfidf_char.shape}",
        "",
        "  ── Word Embeddings (mean-pooled) ──",
        f"  Embedding dim       : {X_train_emb.shape[1]}",
        f"  Source              : {'GloVe' if os.path.exists(GLOVE_PATH) else 'Word2Vec (trained on corpus)'}",
        f"  Normalisation       : L2",
        f"  X_train_embeddings  : {X_train_emb.shape}",
        f"  X_test_embeddings   : {X_test_emb.shape}",
        "",
        "  ── Combined Features (TF-IDF + Embeddings) ──",
        f"  X_train_combined    : {X_train_combined.shape}",
        f"  X_test_combined     : {X_test_combined.shape}",
        "",
        "  ── Labels ──",
        f"  y_train             : {y_train.shape}  — ham: {(y_train==0).sum()}  spam: {(y_train==1).sum()}",
        f"  y_test              : {y_test.shape}   — ham: {(y_test==0).sum()}   spam: {(y_test==1).sum()}",
        f"  y_train_balanced    : {y_train_balanced.shape}",
        "",
        "  ── Saved Files ──",
        f"  {out}/X_train_tfidf.npz",
        f"  {out}/X_test_tfidf.npz",
        f"  {out}/X_train_tfidf_char.npz",
        f"  {out}/X_test_tfidf_char.npz",
        f"  {out}/X_train_embeddings.npy",
        f"  {out}/X_test_embeddings.npy",
        f"  {out}/X_train_combined.npz",
        f"  {out}/X_test_combined.npz",
        f"  {out}/tfidf_vectorizer.pkl",
        f"  {out}/tfidf_char_vectorizer.pkl",
        f"  {out}/y_train.npy",
        f"  {out}/y_test.npy",
        f"  {out}/y_train_balanced.npy",
        "=" * 65,
    ]

    summary_text = "\n".join(summary_lines)
    with open(f"{out}/feature_summary.txt", "w") as fh:
        fh.write(summary_text)

    print("\n" + summary_text)


# ─────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  EMAIL SPAM CLASSIFICATION — FEATURE EXTRACTION")
    print("=" * 65)

    # ── 0. Load preprocessed data ─────────────────────────────────────────
    (
        df_train, df_test,
        X_train_count, X_test_count,
        y_train, y_test, y_train_balanced,
        count_vec,
    ) = load_preprocessed()

    train_texts = df_train["clean_text"].fillna("")
    test_texts  = df_test["clean_text"].fillna("")

    # ── 1. TF-IDF word n-grams ────────────────────────────────────────────
    X_train_tfidf, X_test_tfidf, tfidf_vec = extract_tfidf_word(
        train_texts, test_texts
    )

    # ── 2. TF-IDF character n-grams ───────────────────────────────────────
    X_train_tfidf_char, X_test_tfidf_char, tfidf_char_vec = extract_tfidf_char(
        train_texts, test_texts
    )

    # ── 3. Word embeddings (GloVe or Word2Vec) ────────────────────────────
    X_train_emb, X_test_emb = extract_embeddings(train_texts, test_texts)

    # ── 4. Combined feature matrix ────────────────────────────────────────
    X_train_combined, X_test_combined = combine_features(
        X_train_tfidf, X_test_tfidf,
        X_train_emb,   X_test_emb,
    )

    # ── 5. Save everything ────────────────────────────────────────────────
    save_outputs(
        X_train_tfidf,      X_test_tfidf,       tfidf_vec,
        X_train_tfidf_char, X_test_tfidf_char,  tfidf_char_vec,
        X_train_emb,        X_test_emb,
        X_train_combined,   X_test_combined,
        y_train,            y_test,             y_train_balanced,
    )

    print("\n  ✓ Feature extraction complete.")
    print(f"  All outputs saved to: {os.path.abspath(OUTPUT_DIR)}\n")


# ─────────────────────────────────────────────────────────────────────────
#  USAGE NOTES
# ─────────────────────────────────────────────────────────────────────────
# Run:
#   python feature_extraction.py
#
# Prerequisites:
#   pip install scikit-learn numpy pandas scipy
#   pip install gensim          # only if GloVe file is absent
#
# Optional GloVe download (100-dim, 822 MB):
#   wget http://nlp.stanford.edu/data/glove.6B.zip
#   unzip glove.6B.zip
#   export GLOVE_PATH=glove.6B.100d.txt
#
# INPUT_DIR must contain the 9 artefacts produced by Preprocess.ipynb:
#   train_cleaned.csv, test_cleaned.csv,
#   X_train_count.npz, X_test_count.npz,
#   y_train.npy, y_test.npy, y_train_balanced.npy,
#   count_vectorizer.pkl, vocabulary.csv
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
