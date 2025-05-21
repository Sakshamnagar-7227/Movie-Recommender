from recommender import MovieRecommender
import streamlit as st
from utils import load_data

st.set_page_config(page_title = "Movie Recommender", page_icon = "🎬", layout = "wide")

st.title("Movie Recommender System")

movies_df = load_data()
recommender = MovieRecommender(movies_df)

movies_list = movies_df['Movie_name'].dropna().unique()
selected_movie = st.selectbox("Choose a movie", sorted(movies_list))

if st.button("Recommend"):
    with st.spinner("Generating recommendations..."):
        recommendations = recommender.get_recommendations(selected_movie)
        if recommendations.empty:
            st.error("NO recommendations found.")
        else:
            for _, row in recommendations.iterrows():
                st.subheader(row["Movie_name"])
                st.markdown(f"**Genre:** {row["Genre"]}")
                st.markdown(f"**Metascore:** {row["Metascore"]} &nbsp;&nbsp; **Rating** {row["Rating_from_10"]}")
                st.markdown("---")
