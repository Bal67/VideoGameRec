import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load data
def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
    df['game_title'] = df['game_title'].str.title()  # Capitalize first letter of each word
    df['game_id'] = df['game_title'].astype('category').cat.codes  # Create game_id
    df['user_id'] = df['user_id'].astype('category').cat.codes  # Ensure user_id is a category
    return df

# Prepare data for neural network model
def prepare_data_for_nn(df):
    user_ids = df['user_id'].astype('category').cat.codes.values
    game_ids = df['game_title'].astype('category').cat.codes.values
    ratings = df['rating'].values
    return user_ids, game_ids, ratings

# Recommend top games based on fine-tuned neural network model
def recommend_games_nn(game_title, df, model, num_users, num_games):
    game_id = df[df['game_title'] == game_title]['game_id'].values[0]
    user_ids = np.array([0] * num_games)  # Assuming user_id 0 for recommendations
    game_ids = np.arange(num_games)
    predictions = model.predict([user_ids, game_ids]).flatten()
    
    top_indices = predictions.argsort()[-10:][::-1]  # Get top 10 recommendations
    top_games = df.iloc[top_indices]['game_title'].unique().tolist()
    
    # Remove the input game and ensure diversity
    top_games = [game for game in top_games if game != game_title]
    return top_games[:5]  # Return top 5 diverse recommendations

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
    fine_tuned_model_path = './models/fine_tuned_model.h5'

    df = load_data(dataset_path)
    user_ids, game_ids, ratings = prepare_data_for_nn(df)
    fine_tuned_model = load_model(fine_tuned_model_path)

    game_title = st.selectbox("Select your favorite game", df['game_title'].unique())

    if st.button("Get Recommendations"):
        st.markdown("<div class='stSubtitle'>🌟 Recommendations from Fine-Tuned Neural Network Model 🌟</div>", unsafe_allow_html=True)
        num_users = len(np.unique(user_ids))
        num_games = len(np.unique(game_ids))
        recommendations = recommend_games_nn(game_title, df, fine_tuned_model, num_users, num_games)
        for game in recommendations:
            st.markdown(f"<div class='recommendation-item'>{game}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
