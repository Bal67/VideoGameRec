import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from math import sqrt

# Load Dataset
def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
    return df

def prepare_data_for_knn(df):
    """
    Prepare the data for training the KNN model.
    
    Args:
        df (pd.DataFrame): Dataframe containing user-item interactions and ratings.
    
    Returns:
        csr_matrix: User-item interaction matrix in CSR format.
        list: List of user IDs.
        list: List of game titles.
    """
    try:
        # Aggregate ratings by taking the mean for each user-game combination
        df = df.groupby(['user_id', 'game_title']).rating.mean().reset_index()

        user_game_matrix = df.pivot(index='user_id', columns='game_title', values='rating').fillna(0)
    except KeyError as e:
        print(f"Columns in the dataset: {df.columns.tolist()}")
        raise e
    except ValueError as e:
        print(f"ValueError: {e}")
        print(f"Data causing the issue: {df.head()}")
        raise e

    user_ids = list(user_game_matrix.index)
    game_titles = list(user_game_matrix.columns)
    user_game_matrix_csr = csr_matrix(user_game_matrix.values)
    
    return user_game_matrix_csr, user_ids, game_titles

# Base model created using KNN
def train_base_model(user_game_matrix_csr):
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_game_matrix_csr)
    return knn

def prepare_data_for_nn(df):
    try:
        user_ids = df['user_id'].astype('category').cat.codes.values
        game_ids = df['game_title'].astype('category').cat.codes.values
        ratings = df['rating'].values
    except KeyError as e:
        print(f"Columns in the dataset: {df.columns.tolist()}")
        raise e
    return user_ids, game_ids, ratings

# Naive model created using neural Collaborative Filtering
def build_naive_model(num_users, num_games, embedding_size=50):
    user_input = Input(shape=(1,))
    game_input = Input(shape=(1,))
    
    user_embedding = Embedding(num_users, embedding_size)(user_input)
    game_embedding = Embedding(num_games, embedding_size)(game_input)
    
    user_vector = Flatten()(user_embedding)
    game_vector = Flatten()(game_embedding)
    
    dot_product = Dot(axes=1)([user_vector, game_vector])
    
    output = Dense(1, activation='linear')(dot_product)
    
    model = Model(inputs=[user_input, game_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    
    return model

# Fine-tuned Naive Model
def build_fine_tuned_model(num_users, num_games, embedding_size=50):
    user_input = Input(shape=(1,))
    game_input = Input(shape=(1,))
    
    user_embedding = Embedding(num_users, embedding_size)(user_input)
    game_embedding = Embedding(num_games, embedding_size)(game_input)
    
    user_vector = Flatten()(user_embedding)
    game_vector = Flatten()(game_embedding)
    
    dot_product = Dot(axes=1)([user_vector, game_vector])
    
    dense_1 = Dense(128, activation='relu')(dot_product)
    dropout_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(64, activation='relu')(dropout_1)
    dropout_2 = Dropout(0.5)(dense_2)
    output = Dense(1, activation='linear')(dropout_2)
    
    model = Model(inputs=[user_input, game_input], outputs=output)
    model.compile(optimizer=Adam(lr=0.0001), loss='mse')
    
    return model



# Save model to desired path
def save_model(model, model_name, model_path='./models'):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    model_file = os.path.join(model_path, model_name)
    
    if isinstance(model, NearestNeighbors):
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
    else:
        model.save(model_file)
        
    print(f"Model saved to {model_file}")

def calculate_rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def evaluate_knn_model(knn, user_game_matrix_csr, test_data):
    test_data = test_data.copy()
    test_data['user_idx'] = test_data['user_id'].astype('category').cat.codes
    test_data['game_idx'] = test_data['game_title'].astype('category').cat.codes
    
    user_indices = test_data['user_idx'].values
    game_indices = test_data['game_idx'].values
    true_ratings = test_data['rating'].values
    
    # Predict ratings
    distances, indices = knn.kneighbors(user_game_matrix_csr[user_indices])
    predicted_ratings = []
    for i in range(len(user_indices)):
        neighbors = indices[i]
        neighbor_ratings = user_game_matrix_csr[neighbors, game_indices[i]].toarray().flatten()
        predicted_rating = neighbor_ratings.mean()
        predicted_ratings.append(predicted_rating)
    
    return calculate_rmse(true_ratings, np.array(predicted_ratings))

def evaluate_nn_model(model, user_ids_test, game_ids_test, ratings_test):
    predictions = model.predict([user_ids_test, game_ids_test]).flatten()
    rmse = calculate_rmse(ratings_test, predictions)
    return rmse




if __name__ == "__main__":
    dataset_path = './data/processed_data.csv'
    
    # Load the data
    df = load_data(dataset_path)
    
    # Prepare data for base model
    user_game_matrix_csr, user_ids, game_titles = prepare_data_for_knn(df)
    
    # Split the data into training and test sets
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    # Train base model
    print("Training base model (KNN)...")
    base_model = train_base_model(user_game_matrix_csr)
    save_model(base_model, 'base_model_knn.pkl')
    
    # Evaluate base model
    print("Evaluating base model (KNN)...")
    knn_rmse = evaluate_knn_model(base_model, user_game_matrix_csr, test_data)
    print(f"KNN Model RMSE: {knn_rmse}")
    
    # Prepare data for neural network models
    user_ids, game_ids, ratings = prepare_data_for_nn(df)
    
    # Split the data into training and test sets
    user_ids_train, user_ids_test, game_ids_train, game_ids_test, ratings_train, ratings_test = train_test_split(
        user_ids, game_ids, ratings, test_size=0.2, random_state=42
    )
    
    # Train naive model
    print("Training naive model (Basic Neural Collaborative Filtering)...")
    num_users = len(np.unique(user_ids))
    num_games = len(np.unique(game_ids))
    
    naive_model = build_naive_model(num_users, num_games)
    naive_model.fit([user_ids_train, game_ids_train], ratings_train, epochs=5, batch_size=64, validation_data=([user_ids_test, game_ids_test], ratings_test))
    save_model(naive_model, 'naive_model.h5')
    
    # Evaluate naive model - RMSE
    naive_predictions = naive_model.predict([user_ids_test, game_ids_test]).flatten()
    naive_rmse = calculate_rmse(ratings_test, naive_predictions)
    print(f"Naive Neural Collaborative Filtering Model RMSE: {naive_rmse}")
    
    # Train fine-tuned model
    print("Training fine-tuned model (Enhanced Neural Collaborative Filtering)...")
    fine_tuned_model = build_fine_tuned_model(num_users, num_games)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    fine_tuned_model.fit([user_ids_train, game_ids_train], ratings_train, epochs=20, batch_size=64, validation_data=([user_ids_test, game_ids_test], ratings_test), callbacks=[early_stopping])
    save_model(fine_tuned_model, 'fine_tuned_model.h5')
    
    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model (Enhanced Neural Collaborative Filtering)...")
    fine_tuned_rmse = evaluate_nn_model(fine_tuned_model, user_ids_test, game_ids_test, ratings_test)
    print(f"Fine-Tuned Neural Collaborative Filtering Model RMSE: {fine_tuned_rmse}")
