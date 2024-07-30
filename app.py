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
    pivot = df.pivot_table(index='user_id', columns='game_title', values='rating')
    pivot = pivot.fillna(0)  # Fill NaN values with 0
    pivot = pivot.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)) if (np.max(x) - np.min(x)) != 0 else x, axis=0)
    return pivot

# Compute cosine similarity
def compute_similarity(pivot):
    pivot_sparse = csr_matrix(pivot.values)
    item_similarity = cosine_similarity(pivot_sparse.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=pivot.columns, columns=pivot.columns)
    return item_similarity_df

# Recommend top games based on similarity
def recommend_games(game_title, item_similarity_df):
    if game_title not in item_similarity_df.columns:
        return ["No data available for this game"]
    similar_scores = item_similarity_df[game_title].sort_values(ascending=False)
    top_games = similar_scores.iloc[1:6].index.tolist()  # Get top 5 recommendations, exclude the game itself
    return top_games

# Streamlit app
def main():
    st.set_page_config(page_title="Game Recommendation System", page_icon="🎮")
    
    # Custom CSS for background color and sparkles
    st.markdown(
        """
        <style>
        .stApp {
            background-color: pink;
            background-image: url("https://www.transparenttextures.com/patterns/stardust.png");
            color: black;
        }
        .recommendations {
            background: rgba(255, 255, 255, 0.6);
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0px 0px 10px 2px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🌟 Game Recommendation System 🌟")

    dataset_path = './data/processed_data.csv'
    df = load_data(dataset_path)
    
    pivot = prepare_pivot_table(df)
    item_similarity_df = compute_similarity(pivot)
    
    game_title = st.selectbox("Select your favorite game", item_similarity_df.columns)

    if st.button("Get Recommendations"):
        st.markdown("<div class='recommendations'>", unsafe_allow_html=True)
        st.write("🌟 Recommendations from Item-Based Collaborative Filtering 🌟")
        recommendations = recommend_games(game_title, item_similarity_df)
        for game in recommendations:
            st.write(game)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()