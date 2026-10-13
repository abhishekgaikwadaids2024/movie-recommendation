import pickle
from pathlib import Path

import streamlit as st
import pandas as pd

import os
import pickle
import urllib.request
import streamlit as st

MODEL_URL = (
    "https://github.com/abhishekgaikwadaids2024/"
    "movie-recommendation/releases/download/v1.0.0/similarity.pkl"
)

@st.cache_resource
def load_similarity():
    model_path = "similarity.pkl"

    if not os.path.exists(model_path):
        with st.spinner("Downloading recommendation model..."):
            urllib.request.urlretrieve(MODEL_URL, model_path)

    with open(model_path, "rb") as file:
        return pickle.load(file)

try:
    similarity = load_similarity()
except Exception as e:
    st.error(f"Could not load recommendation model: {e}")
    st.stop()

# ---------------------------------------------------------
# Movie Recommendation System - Streamlit Frontend
# Keep app.py, movie_list.pkl, and similarity.pkl together.
# ---------------------------------------------------------
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background: radial-gradient(ellipse at top left, #20264b 0%, #111322 38%, #0b0c14 78%);
    color: #f6f7fb;
}
.block-container { padding-top: 1.8rem; max-width: 1250px; }
.hero {
    padding: 2rem 2.2rem; border-radius: 24px;
    background: linear-gradient(115deg, rgba(112,75,255,.28), rgba(31,192,177,.10)),
                rgba(255,255,255,.045);
    border: 1px solid rgba(255,255,255,.11);
    margin-bottom: 1.4rem;
}
.eyebrow { color:#9b91ff; font-weight:700; letter-spacing:2px; font-size:.78rem; text-transform:uppercase; }
.hero h1 { font-family:'Outfit',sans-serif; font-size: clamp(2.1rem, 4vw, 3.7rem); line-height:1.05; margin:.4rem 0 .8rem; color:#fff; }
.hero p { color:#c4c8da; font-size:1.03rem; max-width:700px; margin:0; }
.section-title { font-family:'Outfit',sans-serif; font-size:1.5rem; font-weight:700; margin:1.1rem 0 .25rem; color:#fff; }
.muted { color:#aeb3c9; font-size:.92rem; }
.movie-card {
    min-height: 190px; padding: 1.1rem; border-radius: 18px;
    background: linear-gradient(145deg, rgba(255,255,255,.09), rgba(255,255,255,.035));
    border: 1px solid rgba(255,255,255,.10);
    box-shadow: 0 12px 35px rgba(0,0,0,.14);
}
.movie-number { color:#8e83ff; font-size:.8rem; font-weight:700; letter-spacing:1px; }
.movie-title { color:#fff; font-family:'Outfit',sans-serif; font-size:1.15rem; font-weight:700; margin:.55rem 0 .8rem; line-height:1.25; }
.movie-id { color:#aeb3c9; font-size:.78rem; }
div[data-testid="stSelectbox"] label, div[data-testid="stTextInput"] label { color:#e7e8f2 !important; font-weight:600; }
.stButton > button {
    background: linear-gradient(100deg,#7865ff,#4b9dff); color:white; border:0;
    border-radius:12px; font-weight:700; min-height:3rem; width:100%;
}
.stButton > button:hover { border:1px solid #aaa0ff; color:white; }
[data-testid="stSidebar"] { background:rgba(10,11,23,.94); border-right:1px solid rgba(255,255,255,.08); }
hr { border-color:rgba(255,255,255,.12); }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MOVIE_FILE = BASE_DIR / "movie_list.pkl"
SIMILARITY_FILE = BASE_DIR / "similarity.pkl"


@st.cache_resource
def load_model():
    if not MOVIE_FILE.exists() or not SIMILARITY_FILE.exists():
        missing = []
        if not MOVIE_FILE.exists():
            missing.append("movie_list.pkl")
        if not SIMILARITY_FILE.exists():
            missing.append("similarity.pkl")
        raise FileNotFoundError("Missing file(s): " + ", ".join(missing))
    with MOVIE_FILE.open("rb") as f:
        movie_data = pickle.load(f)
    with SIMILARITY_FILE.open("rb") as f:
        similarity_matrix = pickle.load(f)

    if not isinstance(movie_data, pd.DataFrame):
        movie_data = pd.DataFrame(movie_data)
    if "title" not in movie_data.columns:
        raise ValueError("movie_list.pkl must contain a 'title' column.")
    if len(movie_data) != len(similarity_matrix):
        raise ValueError(
            "The number of movies does not match the similarity matrix. "
            "Please use movie_list.pkl and similarity.pkl generated together."
        )
    return movie_data.reset_index(drop=True), similarity_matrix


def get_recommendations(movie_title, movie_data, similarity_matrix, count=5):
    matches = movie_data.index[movie_data["title"].astype(str) == movie_title].tolist()
    if not matches:
        return []
    selected_index = matches[0]
    scores = list(enumerate(similarity_matrix[selected_index]))
    scores.sort(key=lambda item: item[1], reverse=True)
    recommendations = []
    for idx, score in scores:
        if idx == selected_index:
            continue
        row = movie_data.iloc[idx]
        recommendations.append({
            "title": str(row["title"]),
            "movie_id": row.get("movie_id", "—"),
            "score": float(score),
        })
        if len(recommendations) >= count:
            break
    return recommendations


try:
    movies, similarity = load_model()
except Exception as exc:
    st.markdown('<div class="hero"><div class="eyebrow">CINEMATCH · MOVIE DISCOVERY</div><h1>Your next favourite film starts here.</h1><p>A content-based movie recommendation interface powered by your saved model.</p></div>', unsafe_allow_html=True)
    st.error(f"Could not load the recommendation model: {exc}")
    st.info("Put app.py, movie_list.pkl, and similarity.pkl in the same folder, then run:  streamlit run app.py")
    st.stop()

titles = sorted(movies["title"].dropna().astype(str).unique().tolist())
if not titles:
    st.error("No movie titles were found in movie_list.pkl.")
    st.stop()

with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("Your personal movie discovery space")
    st.divider()
    st.markdown("### How it works")
    st.write("Choose a movie you enjoy. The model compares its content tags with other films and ranks the closest matches.")
    recommendation_count = st.slider("Recommendations to show", min_value=3, max_value=10, value=5)
    st.divider()
    st.caption("Model: CountVectorizer + cosine similarity")
    st.caption(f"Movies in library: {len(movies):,}")

st.markdown("""
<div class="hero">
  <div class="eyebrow">✦ YOUR PERSONAL MOVIE GUIDE</div>
  <h1>Find your <span style="color:#9e91ff">next favourite</span> movie.</h1>
  <p>Pick a film you love and discover movies with similar genres, keywords, cast, and directors — recommended by your trained similarity model.</p>
</div>
""", unsafe_allow_html=True)

metric1, metric2, metric3 = st.columns(3)
with metric1:
    st.metric("Movies in catalogue", f"{len(movies):,}")
with metric2:
    st.metric("Recommendation engine", "Content-based")
with metric3:
    st.metric("Model similarity", "Cosine")

st.markdown('<div class="section-title">What do you feel like watching?</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Search or select a movie to get personalised recommendations.</div>', unsafe_allow_html=True)

left, right = st.columns([4, 1])
with left:
    selected_movie = st.selectbox("Choose a movie", options=titles, index=titles.index("The Dark Knight") if "The Dark Knight" in titles else 0)
with right:
    st.write("")
    st.write("")
    recommend_clicked = st.button("✨ Recommend", use_container_width=True)

# Also show recommendations on the initial page, so the UI is not empty.
if recommend_clicked or "selected_movie" not in st.session_state:
    st.session_state["selected_movie"] = selected_movie
else:
    st.session_state["selected_movie"] = selected_movie

st.markdown("---")
st.markdown(f'<div class="section-title">Because you selected <span style="color:#a69bff">{selected_movie}</span></div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Top matches from your saved similarity matrix.</div>', unsafe_allow_html=True)

results = get_recommendations(selected_movie, movies, similarity, recommendation_count)
if not results:
    st.warning("No recommendations found for this title. Try another movie.")
else:
    for start in range(0, len(results), 3):
        chunk = results[start:start+3]
        cols = st.columns(3, gap="medium")
        for offset, result in enumerate(chunk):
            rank = start + offset + 1
            with cols[offset]:
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-number">MATCH #{rank:02d} &nbsp;·&nbsp; {result['score']:.0%} similarity</div>
                    <div class="movie-title">{result['title']}</div>
                    <div class="movie-id">Movie ID: {result['movie_id']}</div>
                </div>
                """, unsafe_allow_html=True)
                st.write("")

with st.expander("ℹ️ About this project"):
    st.write(
        "This interface uses the two pickle files created by your notebook: "
        "`movie_list.pkl` (movie titles and combined tags) and `similarity.pkl` "
        "(cosine-similarity scores). It does not retrain the model."
    )
    st.write("Note: this notebook's saved movie table does not include poster URLs, so the interface uses clean movie cards instead of poster images.")
