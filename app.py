import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

# Load data
def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
    df['game_title'] = df['game_title'].str.title()  # Capitalize first letter of each word
    df['game_id'] = df['game_title'].astype('category').cat.codes  # Create game_id
    df['user_id'] = df['user_id'].astype('category').cat.codes  # Ensure user_id is a category
    return df

# Recommend top games based on embeddings
def recommend_games_embedding(game_title, df, model):
    game_ids = df['game_id'].values
    unique_game_ids = np.unique(game_ids)
    
    game_id = df[df['game_title'] == game_title]['game_id'].values[0]
    game_embedding_layer = model.get_layer('game_embedding')
    game_embedding_weights = game_embedding_layer.get_weights()[0]
    game_embedding = game_embedding_weights[game_id]

    similarities = cosine_similarity([game_embedding], game_embedding_weights)[0]
    similar_game_indices = similarities.argsort()[-11:][::-1]  # Get top 10 similar games

    top_game_ids = [idx for idx in similar_game_indices if idx != game_id]
    top_games = df[df['game_id'].isin(top_game_ids)]['game_title'].unique().tolist()

    return top_games[:5]  # Return top 5 recommendations

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

        .stSubtitle {
            text-align: center;
            color: #ffffff;
            font-size: 2em;
            text-shadow: 1px 1px #000000;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        .recommendation-item {
            font-size: 1.2em;
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
    model_path = './models/fine_tuned_model.h5'

    df = load_data(dataset_path)
    model = load_model(model_path)

    game_title = st.selectbox("Select your favorite game", df['game_title'].unique())

    if st.button("Get Recommendations"):
        st.markdown("<div class='stSubtitle'>🌟 Recommendations from Fine-Tuned Neural Network Model 🌟</div>", unsafe_allow_html=True)
        recommendations = recommend_games_embedding(game_title, df, model)
        for game in recommendations:
            st.markdown(f"<div class='recommendation-item'>{game}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
