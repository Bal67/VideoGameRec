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
    
    # Custom CSS for an extravagant look
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');
        
        .stApp {
            background-color: #ff69b4;
            background-image: url("https://www.transparenttextures.com/patterns/45-degree-fabric-dark.png");
            color: #ffffff;
            font-family: 'Poppins', sans-serif;
        }
        
        .stTitle {
            text-align: center;
            color: #ffffff;
            font-size: 3em;
            text-shadow: 2px 2px #000000;
        }
        
        .recommendations {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0px 0px 15px 5px rgba(255, 105, 180, 0.6);
            animation: sparkle 1s infinite;
        }
        
        @keyframes sparkle {
            0% { box-shadow: 0px 0px 15px 5px rgba(255, 105, 180, 0.6); }
            50% { box-shadow: 0px 0px 25px 10px rgba(255, 255, 255, 0.8); }
            100% { box-shadow: 0px 0px 15px 5px rgba(255, 105, 180, 0.6); }
        }
        
        .recommendation-item {
            font-size: 1.5em;
            margin: 10px 0;
            padding: 10px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.1);
            box-shadow: 0px 0px 15px 5px rgba(255, 105, 180, 0.6);
            color: #ffffff;
            animation: glow 1.5s infinite;
        }
        
        @keyframes glow {
            0% { box-shadow: 0px 0px 10px 2px rgba(255, 105, 180, 0.6); }
            50% { box-shadow: 0px 0px 20px 10px rgba(255, 255, 255, 0.8); }
            100% { box-shadow: 0px 0px 10px 2px rgba(255, 105, 180, 0.6); }
        }
        
        .stButton>button {
            background-color: #ff1493;
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 1.2em;
            cursor: pointer;
        }
        
        .stButton>button:hover {
            background-color: #ff69b4;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='stTitle'>🌟 Game Recommendation System 🌟</h1>", unsafe_allow_html=True)

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
            st.markdown(f"<div class='recommendation-item'>{game}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()