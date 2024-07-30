import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load models
def load_knn_model(path):
    with open(path, 'rb') as file:
        knn_model = pickle.load(file)
    return knn_model

def load_nn_model(path):
    model = load_model(path)
    return model

# Load data
def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
    return df

# Calculate RMSE
def calculate_rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

# Make recommendations using KNN model
def knn_recommend(user_id, knn_model, user_game_matrix_csr, df):
    user_idx = df[df['user_id'] == user_id].index[0]
    distances, indices = knn_model.kneighbors(user_game_matrix_csr[user_idx], n_neighbors=10)
    
    recommendations = []
    for idx in indices.flatten():
        game = df.iloc[idx]['game_title']
        recommendations.append(game)
    return recommendations

# Make recommendations using NN model
def nn_recommend(user_id, game_id, nn_model, num_users, num_games):
    user_array = np.array([user_id] * num_games)
    game_array = np.arange(num_games)
    predictions = nn_model.predict([user_array, game_array]).flatten()
    
    top_indices = predictions.argsort()[-10:][::-1]
    return top_indices

# Streamlit app
def main():
    st.title("Game Recommendation System")

    dataset_path = './data/processed_data.csv'
    knn_model_path = './models/base_model_knn.pkl'
    nn_model_path = './models/fine_tuned_model.h5'

    df = load_data(dataset_path)
    knn_model = load_knn_model(knn_model_path)
    nn_model = load_nn_model(nn_model_path)

    user_id = st.number_input("Enter User ID", min_value=1, max_value=df['user_id'].max())
    recommendation_type = st.selectbox("Select Recommendation Type", ("KNN", "Neural Network"))

    if st.button("Get Recommendations"):
        if recommendation_type == "KNN":
            user_game_matrix_csr, user_ids, game_titles = prepare_data_for_knn(df)
            recommendations = knn_recommend(user_id, knn_model, user_game_matrix_csr, df)
            st.write("Recommended Games:")
            for game in recommendations:
                st.write(game)
        else:
            num_users = len(df['user_id'].unique())
            num_games = len(df['game_title'].unique())
            game_id = st.number_input("Enter Game ID", min_value=1, max_value=num_games)
            top_games = nn_recommend(user_id, game_id, nn_model, num_users, num_games)
            st.write("Recommended Games:")
            for idx in top_games:
                game = df[df['game_id'] == idx]['game_title'].values[0]
                st.write(game)

if __name__ == "__main__":
    main()

