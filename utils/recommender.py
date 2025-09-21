import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def stem(text):
    """
    Applies Porter stemming to each word in the input text.

    Args:
        text (str): The input string to be stemmed.

    Returns:
        str: The stemmed version of the input text.
    """
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# Load data
anime = pd.read_csv('data/anime_data1.csv')

# Keep relevant columns
anime_copy = anime[['MAL ID', 'Name', 'English Name', 'Image', 'Synopsis', 'Genre', 'Themes', 'Demographic', 'Studio', 'Score']]
anime_copy = anime_copy[anime_copy['Synopsis'].notna()]

# Convert columns to string
anime_copy['Genre'] = anime_copy['Genre'].astype(str)
anime_copy['Themes'] = anime_copy['Themes'].astype(str)
anime_copy['Demographic'] = anime_copy['Demographic'].astype(str)
anime_copy['Studio'] = anime_copy['Studio'].astype(str)

# Clean up text columns
for col in ['Genre', 'Themes', 'Demographic', 'Studio']:
    anime_copy[col] = anime_copy[col].apply(lambda x: x.replace("'", "")
                                                     .replace("[", "")
                                                     .replace("]", "")
                                                     .replace(",", ""))

# Create tags column
anime_copy['Tags'] = anime_copy['Synopsis'].astype(str) + anime_copy['Genre'].astype(str)

# Drop unused columns
anime_processed_df = anime_copy.drop(['Themes', 'Demographic', 'Studio', 'Score'], axis=1)

# Clean and preprocess tags
anime_processed_df['Tags'] = anime_processed_df['Tags'].apply(lambda x: x.replace("[Written by MAL Rewrite]", ''))
anime_processed_df['Tags'] = anime_processed_df['Tags'].apply(lambda x: x.lower())

ps = PorterStemmer()

anime_processed_df['Tags'] = anime_processed_df['Tags'].apply(stem)

cv = CountVectorizer(max_features=6000, stop_words='english')
vectors = cv.fit_transform(anime_processed_df['Tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(anime, df=anime_processed_df):
    """
    Recommends similar anime titles based on the provided anime name (Japanese or English).

    Args:
        anime (str): The name of the anime to find recommendations for.
        df (pd.DataFrame, optional): DataFrame containing anime information. Defaults to anime_processed_df.

    Returns:
        list: A list of up to 10 recommended anime titles (English names if available).
    
    Raises:
        ValueError: If the anime is not found in the DataFrame.
    """
    # Try to find by Japanese name first
    if anime in df['Name'].values:
        anime_index = df[df['Name'] == anime].index[0]
    # Otherwise try English name
    elif anime in df['English Name'].values:
        anime_index = df[df['English Name'] == anime].index[0]
    else:
        raise ValueError("Anime not found.")

    anime_list = sorted(
        list(enumerate(set(similarity[anime_index]))),
        reverse=True,
        key=lambda x: x[1]
    )

    recommendations = []
    seen_names = set()
    for idx, score in anime_list:
        name = df.iloc[idx]['Name']
        if name != anime and name not in seen_names:
            recommendations.append((idx, score))
            seen_names.add(name)
        if len(recommendations) == 10:
            break

    results = []
    posters = []
    for i in recommendations:
        row = df.iloc[i[0]]
        if type(row['English Name']) == float:
            results.append(row['Name'])
        else:
            results.append(f"{row['English Name']}")
        posters.append(row['Image'])
    
    return results, posters

# Example usage:
# recos, posters = recommend('Steins;Gate', anime_processed_df)  # Example with English name
# print (posters)
