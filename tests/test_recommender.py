import pytest
import pandas as pd
from recommender import recommend, new_df

def test_anime_not_found():
    with pytest.raises(ValueError, match="Anime not found."):
        recommend("Nonexistent Anime Name")

def test_recommend_japanese_name():
    results = recommend("Kimetsu no Yaiba")
    assert len(results) > 0

def test_recommend_english_name():
    results = recommend("The Promised Neverland")
    assert len(results) > 0

def test_no_self_in_recommendations():
    anime_name = "Steins;Gate"
    results = recommend(anime_name)
    for line in results:
        assert anime_name not in line

def test_no_duplicates_in_recommendations():
    anime_name = "Vinland Saga"
    results = recommend(anime_name)
    recommended_names = [line.split('(')[0].strip() for line in results]
    assert len(recommended_names) == len(set(recommended_names))