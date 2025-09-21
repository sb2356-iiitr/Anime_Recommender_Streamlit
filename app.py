import streamlit as st
import pandas as pd
from utils.recommender import recommend, anime_processed_df

st.title("Anime Recommender System")
st.write("Get recommendations for anime titles based on your favorite anime!")

# Combine Japanese and English names for selectbox options
anime_names = pd.concat([
    anime_processed_df['Name'],
    anime_processed_df['English Name']
]).dropna().unique()
options = [""] + list(anime_names)  # Add empty string as default
anime_name = st.selectbox("Enter an anime name (Japanese or English):", options, index=0)

if st.button("Recommend"):
    try:
        # Find the row for the selected anime
        anime_row = anime_processed_df[
            (anime_processed_df['Name'] == anime_name) | 
            (anime_processed_df['English Name'] == anime_name)
        ].iloc[0]
        st.header("You searched for")
        st.image(anime_row['Image'], width=200)
        st.write(anime_name)
        recommendations, posters = recommend(anime_name, anime_processed_df)
        st.header("Recommended Anime Titles:")
        # Arrange recommendations in a 2x5 grid
        cols = st.columns(3)
        for i in range(10):
            col = cols[i % 3]
            with col:
                if i < len(posters):
                    st.image(posters[i], width=200)
                    st.write(recommendations[i])
    except ValueError as e:
        st.error(str(e))


