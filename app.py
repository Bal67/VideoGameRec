import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the fine-tuned model
def load_nn_model(path):
    model = load_model(path)
    return model

# Load data
def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
    df['game_title'] = df['game_title'].str.title()  # Capitalize first letter of each word
    return df

# Calculate RMSE
def calculate_rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

# Make recommendations using the fine-tuned NN model
def nn_recommend(game_title, df, nn_model, num_users, num_games):
    game_id = df[df['game_title'] == game_title].index[0]
    user_ids = np.array([0] * num_games)
    game_ids = np.arange(num_games)
    predictions = nn_model.predict([user_ids, game_ids]).flatten()
    
    top_indices = predictions.argsort()[-20:][::-1]  # Get top 20 recommendations for diversity
    top_games = df.iloc[top_indices]['game_title'].tolist()
    
    # Remove the input game and ensure diversity
    top_games = [game for game in top_games if game != game_title]
    top_games = list(dict.fromkeys(top_games))  # Remove duplicates while preserving order
    
    return top_games[:10]  # Return top 10 diverse recommendations

# Streamlit app
def main():
    st.title("Game Recommendation System")

    dataset_path = './data/processed_data.csv'
    fine_tuned_nn_model_path = './models/fine_tuned_model.h5'

    df = load_data(dataset_path)
    fine_tuned_nn_model = load_nn_model(fine_tuned_nn_model_path)

    game_title = st.selectbox("Select your favorite game", df['game_title'].unique())

    if st.button("Get Recommendations"):
        st.write("Recommendations from Fine-Tuned Neural Network Model:")
        num_users = len(df['user_id'].unique())
        num_games = len(df['game_title'].unique())
        fine_tuned_nn_recommendations = nn_recommend(game_title, df, fine_tuned_nn_model, num_users, num_games)
        for game in fine_tuned_nn_recommendations:
            st.write(game)

if __name__ == "__main__":
    main()
