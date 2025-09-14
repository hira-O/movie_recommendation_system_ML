import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Page Config ----------
st.set_page_config(page_title="Movie Recommender 🎬", page_icon="🎬", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
        .top-bar {
            background-color: #0A2342;
            color: white;
            padding: 12px;
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .filter-box {
            background-color: #F5F5F5;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #DDD;
        }
        .stNumberInput label, .stSlider label, .stSelectbox label, .stTextInput label {
            font-weight: bold;
            color: #0A2342;
        }
        div.stButton > button {
            background-color: #0A2342;
            color: white;
            border-radius: 5px;
            padding: 8px 20px;
            margin-top: 10px;
        }
        div.stButton > button:hover {
            background-color: #1E3A70;
            color: white;
        }
        .blue-table {
            background-color: white;
            border: 1px solid #0A2342;
            border-radius: 8px;
            padding: 10px;
        }
        .blue-table th {
            background-color: #0A2342 !important;
            color: white !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Data ----------
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Extract year from title
movies["year"] = movies["title"].str.extract(r'\((\d{4})\)').astype(float)

# Create user-item matrix
user_item = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
user_sim = cosine_similarity(user_item)
user_sim = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)

# ---------- Fake User Mapping (Name → ID) ----------
user_mapping = {
    "Hira": 1,
    "Ali": 2,
    "Rehan": 3,
    "Shiza": 4
    # aur names add kar sakti ho
}

# ---------- Recommendation Function ----------
def recommend_user_based(user_id, fetch_n=300):
    sim_scores = user_sim.loc[user_id].values
    ratings_other = user_item.values.T.dot(sim_scores)
    sim_sums = (user_item.values.T != 0).dot(sim_scores)

    preds = ratings_other / (sim_sums + 1e-9)
    preds = pd.Series(preds, index=user_item.columns)

    seen = ratings[ratings.userId == user_id]["movieId"].tolist()
    preds = preds.drop(seen, errors="ignore")

    top_movies = preds.sort_values(ascending=False).head(fetch_n).index
    result = movies[movies["movieId"].isin(top_movies)].copy()
    result["score"] = preds[top_movies].values
    return result.sort_values("score", ascending=False)

# ---------- UI ----------
st.markdown('<div class="top-bar">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)

# Step 1: User enters name
user_name = st.text_input("👤 Please enter your name to continue:")

if user_name:
    if user_name not in user_mapping:
        st.error("❌ User not found! Please enter a valid name.")
    else:
        user_id = user_mapping[user_name]

        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown('<div class="filter-box">', unsafe_allow_html=True)

            top_n = st.slider("📌 Number of Movies", min_value=3, max_value=15, value=5)

            genre_filter = st.selectbox("🎭 Genre", ["All"] + sorted(set("|".join(movies["genres"]).split("|"))))
            min_rating = st.slider("⭐ Minimum Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.5)
            year_range = st.slider("📅 Year Range",
                                   int(movies["year"].min(skipna=True)),
                                   int(movies["year"].max(skipna=True)),
                                   (1990, 2010))
            search_text = st.text_input("🔍 Search by Movie Title", "")

            apply = st.button("Apply Filters")
            clear = st.button("Clear Filters")

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.subheader(f" Top {top_n} Recommendations for {user_name}")

            if apply:
                recs = recommend_user_based(user_id, fetch_n=300)

                if genre_filter != "All":
                    recs = recs[recs["genres"].str.contains(genre_filter, na=False)]

                recs = recs[(recs["year"] >= year_range[0]) & (recs["year"] <= year_range[1])]
                recs = recs[recs["score"] >= min_rating]

                if search_text:
                    recs = recs[recs["title"].str.contains(search_text, case=False, na=False)]

                recs = recs.head(top_n)

                if not recs.empty:
                    st.markdown('<div class="blue-table">', unsafe_allow_html=True)
                    st.table(recs[["title", "genres", "year", "score"]].reset_index(drop=True))
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("No movies found matching your filters.")

            elif clear:
                st.info("Filters cleared. Please set filters and click Apply again.")
            else:
                st.info("Set filters and click **Apply Filters** to see recommendations.")
