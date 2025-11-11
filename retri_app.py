# filename: app_news_ir_label_stoplist_feedback_improved.py
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# -------------------- Defaults --------------------
DEFAULT_DATA_DIR = Path("dataset")
DEFAULT_LABEL = Path("label.csv")
DEFAULT_STOP = Path("stoplist.txt")
FEEDBACK_FILE = "feedback.csv"

# -------------------- Util preprocessing --------------------
_TOKEN_RX = re.compile(r"[a-z]+")
def _good_token(tok: str) -> bool:
    return 2 <= len(tok) <= 25 and tok.isalpha()

@st.cache_resource
@st.cache_data
def get_stemmer():
    return StemmerFactory().create_stemmer()

def read_stoplist(path: Path) -> List[str]:
    base_id = {
        "yang","dan","di","ke","dari","untuk","pada","adalah","ini","itu","sebagai","dengan",
        "atau","namun","lebih","oleh","saat","telah","setelah","sebelum","antara","tentang",
        "dalam","mereka","kita","kamu","saya","ia","dia","hingga","agar","bahwa","pun","bagi",
        "secara","misalnya","yakni","jadi","setiap","tiap"
    }
    base_en = {"the","a","an","of","and","to","in","on","for","with","as","by","is","are","was","were","be","been"}
    if path.exists():
        words = [w.strip().lower() for w in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
        words = [w for w in words if w and not w.startswith("#")]
        return sorted(set(words) | base_id | base_en)
    return sorted(base_id | base_en)

def preprocess_text_cached(text: str, stemmer, stopset: set, cache: Dict[str, str]) -> str:
    # Single-word cache based stemming to speed up large corpora
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    t = text.lower()
    toks = _TOKEN_RX.findall(t)
    toks = [w for w in toks if w not in stopset and _good_token(w)]
    out = []
    for w in toks:
        v = cache.get(w)
        if v is None:
            try:
                v = stemmer.stem(w)
            except Exception:
                v = w
            cache[w] = v
        if v and v not in stopset:
            out.append(v)
    return " ".join(out)

# -------------------- I/O helpers --------------------
def norm_stem(name: str) -> str:
    stem = Path(name).stem.lower().strip().replace("\ufeff","")
    return f"data{stem}" if re.fullmatch(r"\d+", stem) else stem

def _read_label_csv_any_format(path: Path) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"Tidak menemukan {path.resolve()}")
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, header=None, names=["filename","category"])
    if df.shape[0] == 0 or len(df.columns) == 1:
        df = pd.read_csv(path, header=None, names=["filename","category"])
    return df

def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, names=["filename", "category"], header=None)
    df["stem"] = df["filename"].astype(str).str.strip()
    return df[["stem", "category"]]


def load_corpus(data_dir: Path) -> pd.DataFrame:
    if not data_dir.exists(): raise FileNotFoundError(f"Folder dataset tidak ditemukan: {data_dir.resolve()}")
    files = [p for p in sorted(data_dir.iterdir()) if p.is_file() and p.name.lower().startswith("data")]
    if not files: raise FileNotFoundError(f"Tidak ada berkas 'data*' di {data_dir.resolve()}")
    recs = [{"stem": norm_stem(p.name), "filename": p.name, "path": str(p.resolve()),
             "text": p.read_text(encoding="utf-8", errors="ignore")} for p in files]
    return pd.DataFrame(recs)

def find_missing(df_docs: pd.DataFrame, df_labels: pd.DataFrame) -> Tuple[List[str], List[str]]:
    docs = set(df_docs["stem"]); labs = set(df_labels["stem"])
    rx = lambda s: int(re.search(r"\d+", s).group()) if re.search(r"\d+", s) else 1e9
    missing = sorted(docs - labs, key=rx)
    extras  = sorted(labs - docs, key=rx)
    return missing, extras

# -------------------- Feedback persistence --------------------
def ensure_feedback_file(path: str = FEEDBACK_FILE):
    if not os.path.exists(path):
        pd.DataFrame(columns=["filename", "score", "count"]).to_csv(path, index=False)

def read_feedback(path: str = FEEDBACK_FILE) -> pd.DataFrame:
    ensure_feedback_file(path)
    return pd.read_csv(path)

def get_feedback_scores_dict(path: str = FEEDBACK_FILE) -> Dict[str, float]:
    df = read_feedback(path)
    if df.empty:
        return {}
    # Use diminishing returns: scaled = sign(score) * (1 - 1/(1+abs(score)))
    scaled = {}
    for _, r in df.iterrows():
        s = float(r.get("score", 0.0))
        # scaled value in (-1,1)
        scaled_val = np.sign(s) * (1 - 1/(1 + abs(s)))
        scaled[r["filename"]] = scaled_val
    return scaled

def update_feedback_record(filename: str, vote: str, path: str = FEEDBACK_FILE):
    ensure_feedback_file(path)
    df = pd.read_csv(path)
    if filename not in df["filename"].values:
        df.loc[len(df)] = [filename, 0.0, 0]
    idx = df.index[df["filename"] == filename][0]
    current = float(df.loc[idx, "score"])
    cnt = int(df.loc[idx, "count"])
    delta = 1.0 if vote == "up" else -1.0
    # adaptive increment: larger count -> smaller added absolute effect on raw score
    adjusted_delta = delta / (1 + 0.2 * cnt)
    df.loc[idx, "score"] = current + adjusted_delta
    df.loc[idx, "count"] = cnt + 1
    df.to_csv(path, index=False)

# -------------------- IR System (Hybrid TF-IDF + centroid subspace) --------------------
class IRSystem:
    def __init__(self, df_docs: pd.DataFrame, df_labels: pd.DataFrame, stopwords: List[str], stemmer):
        # merge docs & labels
        merged = df_docs.merge(df_labels, on="stem", how="left")
        if merged["category"].isna().any():
            merged.loc[merged["category"].isna(), "category"] = "Unknown"
            st.warning("Ada dokumen tanpa label → ditandai 'Unknown'")
        self.df = merged.reset_index(drop=True)
        self.stemmer = stemmer
        self.stopset = set(stopwords)

        # preprocessing with single-word caching
        st.info("Preprocessing (Sastrawi + stoplist) ...")
        stem_cache: Dict[str, str] = {}
        proc_texts = []
        pbar = st.progress(0, text="Menstem dokumen...")
        total = len(self.df)
        for i, txt in enumerate(self.df["text"].tolist(), 1):
            proc_texts.append(preprocess_text_cached(txt, self.stemmer, self.stopset, stem_cache))
            if i % 5 == 0 or i == total:
                pbar.progress(i/total, text=f"Menstem dokumen... {i}/{total}")
        pbar.empty()
        self.df["proc_text"] = proc_texts

        # TF-IDF vectorizer (l2 normalized)
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, norm="l2", min_df=1)
        self.X = self.vectorizer.fit_transform(self.df["proc_text"].fillna(""))

        # labels and centroids (centroid computed in TF-IDF space)
        self.labels = self.df["category"].astype(str)
        self.cats = sorted(self.labels.unique())
        self.centroids = self._build_centroids()

    def _build_centroids(self) -> Dict[str, np.ndarray]:
        cdict: Dict[str, np.ndarray] = {}
        for c in self.cats:
            idx = np.where(self.labels.values == c)[0]
            if len(idx) == 0:
                continue
            centroid_dense = np.asarray(self.X[idx].mean(axis=0)).ravel()
            # if centroid is all zeros, skip
            if centroid_dense.sum() == 0:
                continue
            cdict[c] = centroid_dense
        return cdict

    def infer_categories(self, query: str, top_k: int = 2, threshold: float = 0.10, margin: float = 0.02):
        stem_cache: Dict[str, str] = {}
        qproc = preprocess_text_cached(query, self.stemmer, self.stopset, stem_cache)
        qv = self.vectorizer.transform([qproc])
        if qv.nnz == 0 or not self.centroids:
            return ["Unknown"], {c: 0.0 for c in self.centroids}
        sims = {c: float(cosine_similarity(qv, self.centroids[c].reshape(1, -1)).ravel()[0]) for c in self.centroids}
        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        chosen = [ranked[0][0]] if ranked else []
        if (not ranked) or ranked[0][1] < threshold or (len(ranked) > 1 and (ranked[0][1] - ranked[1][1]) < margin):
            chosen = [c for c, _ in ranked[:max(1, top_k)]]
        return chosen, {k: round(v, 4) for k, v in sims.items()}

    def retrieve(self, query: str, topn: int = 10, feedback_alpha: float = 0.10):
        chosen_cats, cat_sims = self.infer_categories(query)
        mask = self.labels.isin(chosen_cats).values
        sub_idx = np.where(mask)[0]
        if len(sub_idx) == 0:
            sub_idx = np.arange(self.X.shape[0])

        # query vector in TF-IDF space
        stem_cache: Dict[str, str] = {}
        qproc = preprocess_text_cached(query, self.stemmer, self.stopset, stem_cache)
        qv = self.vectorizer.transform([qproc])

        # cosine similarities
        scores = cosine_similarity(qv, self.X[sub_idx]).ravel()  # already l2 normalized from TF-IDF
        if scores.size == 0:
            return pd.DataFrame(columns=["rank","filename","category","path","text","score"])

        # integrate feedback (scaled, in [-1,1])
        feedback_scores = get_feedback_scores_dict()
        fb_array = np.array([feedback_scores.get(self.df.loc[idx, "filename"], 0.0) for idx in sub_idx], dtype=float)
        # normalize both arrays to [0,1] for stable linear mix
        def norm01(arr):
            if arr.size == 0:
                return arr
            mn, mx = arr.min(), arr.max()
            if mx - mn <= 1e-9:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)
        scores_n = norm01(scores)
        fb_n = norm01(fb_array)
        # weighted combination
        final = (1 - feedback_alpha) * scores_n + feedback_alpha * fb_n

        order = final.argsort()[::-1][:topn]
        rows = sub_idx[order]
        out = self.df.iloc[rows][["filename","category","path","text"]].copy()
        out.insert(0, "rank", np.arange(1, len(order)+1))
        out["score"] = np.round(final[order], 4)
        out = out.reset_index(drop=True)
        return out

# -------------------- Rerank helpers (keep original UX) --------------------
def rerank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df["rank"] = np.arange(1, len(df) + 1)
    return df

def reverse_results(df: pd.DataFrame) -> pd.DataFrame:
    return rerank(df.iloc[::-1].reset_index(drop=True))

def demote_by_id(df: pd.DataFrame, article_id: str) -> pd.DataFrame:
    idx = df.index[df["path"] == article_id].tolist()
    if not idx: return df
    row = df.iloc[[idx[0]]].copy()
    df2 = pd.concat([df.drop(index=idx[0]), row], ignore_index=True)
    return rerank(df2)

def demote_by_rank(df: pd.DataFrame, rank: int) -> pd.DataFrame:
    if rank < 1 or rank > len(df): return df
    row = df.iloc[[rank-1]].copy()
    df2 = pd.concat([df.drop(index=rank-1), row], ignore_index=True)
    return rerank(df2)

# -------------------- Streamlit UI (preserve UX & features) --------------------
st.set_page_config(page_title="News IR (improved)", layout="wide")
st.title("📰 News IR — Hybrid TF-IDF + Context Centroid + Feedback")

st.sidebar.header("⚙️ Pengaturan")
data_dir = Path(st.sidebar.text_input("Folder dataset", str(DEFAULT_DATA_DIR)))
label_csv = Path(st.sidebar.text_input("File label.csv", str(DEFAULT_LABEL)))
stoplist_txt = Path(st.sidebar.text_input("File stoplist.txt (opsional)", str(DEFAULT_STOP)))
topn = st.sidebar.slider("Top-N hasil", 5, 50, 10, 1)
feedback_alpha = st.sidebar.slider("Bobot feedback terhadap ranking (α)", 0.0, 0.5, 0.10, 0.01)
st.sidebar.caption("Hybrid: centroid-based subspace selection + TF-IDF ranking. Feedback mempengaruhi urutan secara lembut.")

# load corpus + labels + IR
if "ir" not in st.session_state or st.sidebar.button("Muat ulang korpus"):
    try:
        df_docs = load_corpus(data_dir)
        df_labels = load_labels(label_csv)
        stopwords = read_stoplist(stoplist_txt)
        stemmer = get_stemmer()
        st.session_state["df_docs"] = df_docs
        st.session_state["df_labels"] = df_labels
        st.session_state["stopwords"] = stopwords
        st.session_state["stemmer"] = stemmer
        st.session_state["ir"] = IRSystem(df_docs, df_labels, stopwords, stemmer)
        st.session_state["results"] = None
        st.session_state["last_read_id"] = None
        ensure_feedback_file()
        st.success("Korpus dimuat.")
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")

ir: Optional[IRSystem] = st.session_state.get("ir")

with st.expander("🔎 Cek dokumen yang belum punya label", expanded=False):
    if "df_docs" in st.session_state and "df_labels" in st.session_state:
        missing, extras = find_missing(st.session_state["df_docs"], st.session_state["df_labels"])
        if not missing:
            st.success("Semua file di dataset sudah ada di label.csv ✅")
        else:
            nums = [re.search(r"\d+", s).group() for s in missing if re.search(r"\d+", s)]
            st.warning(f"Belum dilabeli: data{', data'.join(nums)}")
            st.dataframe(pd.DataFrame({"stem": missing}), use_container_width=True)
        if extras:
            st.info(f"Label tanpa file di dataset: {', '.join(extras)}")

st.subheader("🔍 Pencarian")
with st.form("search"):
    query = st.text_input("Ketik kueri (ID):", "")
    if st.form_submit_button("Cari") and ir:
        st.session_state["results"] = ir.retrieve(query, topn=topn, feedback_alpha=feedback_alpha)
        st.session_state["last_read_id"] = None

results: Optional[pd.DataFrame] = st.session_state.get("results")

colg1, colg2, colg3 = st.columns([1,1,2])
with colg1:
    if st.button("🔄 Jempol atas (balik urutan)", use_container_width=True, disabled=results is None):
        st.session_state["results"] = reverse_results(results)
with colg2:
    if st.button("👇 Jempol bawah (terakhir dibaca)", use_container_width=True, disabled=st.session_state.get("last_read_id") is None):
        st.session_state["results"] = demote_by_id(st.session_state["results"], st.session_state["last_read_id"])
        st.session_state["last_read_id"] = None
with colg3:
    with st.popover("Perintah teks"):
        cmd = st.text_input("Contoh: 'jempol bawah', 'jempol bawah 5', 'jempol atas'")
        if st.button("Jalankan"):
            low = cmd.strip().lower()
            if low in {"jempol atas","balik"} and results is not None:
                st.session_state["results"] = reverse_results(results)
            elif (m := re.match(r"^jempol\s+bawah\s+(\d+)$", low)) and results is not None:
                st.session_state["results"] = demote_by_rank(results, int(m.group(1)))
            elif low == "jempol bawah":
                lrid = st.session_state.get("last_read_id")
                if lrid and results is not None:
                    st.session_state["results"] = demote_by_id(results, lrid)
                    st.session_state["last_read_id"] = None
                else:
                    st.warning("Belum ada artikel yang dibaca. Buka artikel dulu, atau pakai 'jempol bawah N'.")
            else:
                st.info("Perintah tidak dikenali.")

st.subheader("📄 Hasil Pencarian")
if results is None:
    st.info("Masukkan kueri untuk menampilkan hasil.")
else:
    cats_counts = results["category"].value_counts().to_dict()
    st.caption("Distribusi kategori: " + ", ".join(f"{k}={v}" for k,v in cats_counts.items()))
    st.dataframe(results[["rank","filename","category","score"]], use_container_width=True, hide_index=True)

    st.divider()
    st.write("### Baca & Kelola Hasil")
    for i, row in results.iterrows():
        with st.expander(f"[{int(row['rank'])}] {row['filename']} • {row['category']} • skor={row['score']}", expanded=False):
            preview_len = st.slider("Panjang preview", 100, 1000, 300, 50, key=f"pv_{i}")
            st.write(row["text"][:preview_len] + ("..." if len(row["text"]) > preview_len else ""))

            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                if st.button("📖 Baca lengkap", key=f"read_{i}"):
                    st.session_state["last_read_id"] = row["path"]
                    st.toast(f"Membuka {row['filename']}")
                    st.markdown(str(row["text"]).encode('utf-8').decode('unicode_escape').replace("\n", "\n\n"))


            with c2:
                if st.button("👎 Jempol bawah (artikel ini)", key=f"down_{i}"):
                    # demote visually now and record feedback (will affect next retrieval)
                    st.session_state["results"] = demote_by_id(st.session_state["results"], row["path"])
                    update_feedback_record(row["filename"], "down")
                    st.rerun()
            with c3:
                if st.button("👍 Jempol atas (artikel ini)", key=f"up_{i}"):
                    # reward slightly (record) — this will impact next retrieval
                    update_feedback_record(row["filename"], "up")
                    st.success("Terima kasih! Skor akan mempengaruhi pencarian berikutnya.")
                st.caption(f"Path: {row['path']}")

    if st.session_state.get("last_read_id"):
        try:
            last = results.loc[results["path"] == st.session_state["last_read_id"]].iloc[0]
            with st.expander("🕮 Terakhir dibaca", expanded=False):
                st.write(f"**{last['filename']}** • {last['category']} • skor={last['score']}")
                st.write(last["text"])
                c1, c2 = st.columns([1,1])
                with c1:
                    if st.button("👎 Jempol bawah (terakhir dibaca)", key="last_down"):
                        st.session_state["results"] = demote_by_id(st.session_state["results"], last["path"])
                        update_feedback_record(last["filename"], "down")
                        st.session_state["last_read_id"] = None
                        st.rerun()
                with c2:
                    if st.button("👍 Jempol atas (terakhir dibaca)", key="last_up"):
                        update_feedback_record(last["filename"], "up")
                        st.success("Terima kasih! Skor akan mempengaruhi pencarian berikutnya.")
        except Exception:
            pass

st.write("---")
st.caption("Hybrid: centroid subspace untuk context recognition; TF-IDF (l2) untuk ranking; feedback scaled & mixed (α).")
