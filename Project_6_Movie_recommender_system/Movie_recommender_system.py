import ast
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MOVIES_CSV = "tmdb_5000_movies.csv"
CREDITS_CSV = "tmdb_5000_credits.csv"

OUTPUT_DIR = "."  # where movie_list.pkl / similarity.pkl will be saved

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i["name"])
    return L


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i["name"])
        counter += 1
    return L


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i["job"] == "Director":
            L.append(i["name"])
    return L


def collapse(L):
    return [i.replace(" ", "") for i in L]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_recommender():
    if not os.path.exists(MOVIES_CSV) or not os.path.exists(CREDITS_CSV):
        raise FileNotFoundError(
            f"Could not find '{MOVIES_CSV}' and/or '{CREDITS_CSV}'.\n"
            "Download the TMDB 5000 dataset from Kaggle and place the two "
            "CSV files next to this script, or update MOVIES_CSV / CREDITS_CSV "
            "at the top of the file."
        )

    print("Loading data...")
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')

    # Merge on title
    movies = movies.merge(credits, on="title")

    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

    movies.dropna(inplace=True)

    print("Parsing genres, keywords, cast, and crew...")
    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert)
    movies["cast"] = movies["cast"].apply(lambda x: x[0:3]) 
    movies["crew"] = movies["crew"].apply(fetch_director)
    
    movies["cast"] = movies["cast"].apply(collapse)
    movies["crew"] = movies["crew"].apply(collapse)
    movies["genres"] = movies["genres"].apply(collapse)
    movies["keywords"] = movies["keywords"].apply(collapse)

    # Split overview into a list of words
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    movies["tags"] = (
        movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
    )

    new = movies.drop(columns=["overview", "genres", "keywords", "cast", "crew"])
    new["tags"] = new["tags"].apply(lambda x: " ".join(x))

    print("Vectorizing tags...")
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vector = cv.fit_transform(new["tags"]).toarray()

    print("Computing cosine similarity matrix (this may take a moment)...")
    similarity = cosine_similarity(vector)

    return new, similarity


def recommend(movie_title, new, similarity, top_n=5):
    matches = new[new["title"] == movie_title]
    if matches.empty:
        print(f"'{movie_title}' not found in the dataset.")
        return

    index = matches.index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    print(f"\nMovies similar to '{movie_title}':")
    for i in distances[1 : top_n + 1]:
        print(" -", new.iloc[i[0]].title)
        
if __name__ == "__main__":
    new, similarity = build_recommender()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "movie_list.pkl"), "wb") as f:
        pickle.dump(new, f)
    with open(os.path.join(OUTPUT_DIR, "similarity.pkl"), "wb") as f:
        pickle.dump(similarity, f)
    print(f"\nSaved 'movie_list.pkl' and 'similarity.pkl' to '{OUTPUT_DIR}'.")
    
    movie_name = input("\nEnter a movie title: ").strip()
    recommend(movie_name, new, similarity)