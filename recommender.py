import numpy as np
import pandas as pd
import os

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
new_df = anime_copy.drop(['Themes', 'Demographic', 'Studio', 'Score'], axis=1)

# Clean and preprocess tags
new_df['Tags'] = new_df['Tags'].apply(lambda x: x.replace("[Written by MAL Rewrite]", ''))
new_df['Tags'] = new_df['Tags'].apply(lambda x: x.lower())

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['Tags'] = new_df['Tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=6000, stop_words='english')
vectors = cv.fit_transform(new_df['Tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

def recommend(anime, df=new_df):
    """
    Recommend similar anime based on Japanese name or English name.
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
    for i in recommendations:
        row = df.iloc[i[0]]
        if type(row['English Name']) == float:
            results.append(row['Name'])
        else:
            results.append(f"{row['English Name']}")
    
    return results

# Example usage:
recos = recommend('Steins;Gate', new_df)  # Example with English name
print (recos)