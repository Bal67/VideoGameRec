import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Load data
def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
    df['game_title'] = df['game_title'].str.title()  # Capitalize first letter of each word
    df['game_id'] = df['game_title'].astype('category').cat.codes  # Create game_id
    return df

# Create pivot table and normalize data
def prepare_pivot_table(df):
    pivot = df.pivot_table(index='game_id', columns='game_title', values='rating')
    pivot = pivot.fillna(0)  # Fill NaN values with 0
    pivot = pivot.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)) if (np.max(x) - np.min(x)) != 0 else x, axis=1)
    pivot = pivot.loc[:, (pivot != 0).any(axis=0)]
    return pivot

# Compute cosine similarity
def compute_similarity(pivot):
    pivot_sparse = csr_matrix(pivot.values)
    item_similarity = cosine_similarity(pivot_sparse)
    item_similarity_df = pd.DataFrame(item_similarity, index=pivot.index, columns=pivot.index)
    return item_similarity_df

# Recommend top games based on similarity
def recommend_games(game_title, item_similarity_df, df):
    game_id = df[df['game_title'] == game_title]['game_id'].values[0]
    similar_scores = item_similarity_df.loc[game_id].sort_values(ascending=False)
    top_game_ids = similar_scores.iloc[1:11].index.tolist()  # Exclude the game itself
    top_games = df[df['game_id'].isin(top_game_ids)]['game_title'].unique().tolist()
    return top_games

# Streamlit app
def main():
    st.title("Game Recommendation System")

    dataset_path = './data/processed_data.csv'
    df = load_data(dataset_path)
    
    pivot = prepare_pivot_table(df)
    item_similarity_df = compute_similarity(pivot)
    
    game_title = st.selectbox("Select your favorite game", pivot.columns)

    if st.button("Get Recommendations"):
        st.write("Recommendations from Item-Based Collaborative Filtering:")
        recommendations = recommend_games(game_title, item_similarity_df, df)
        for game in recommendations:
            st.write(game)

if __name__ == "__main__":
    main()
