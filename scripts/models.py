import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from math import sqrt

# Load Dataset
def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
    return df

# Prepare dataset for KNN
def prepare_data_for_knn(df):
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

def prepare_data_for_nn(df):
    user_ids = df['user_id'].astype('category').cat.codes.values
    game_ids = df['game_title'].astype('category').cat.codes.values
    ratings = df['rating'].values
    return user_ids, game_ids, ratings

# Base model created using KNN
def train_base_model(user_game_matrix_csr):
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_game_matrix_csr)
    return knn

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
    
    user_embedding = Embedding(num_users, embedding_size, embeddings_regularizer='l2', name='user_embedding')(user_input)
    game_embedding = Embedding(num_games, embedding_size, embeddings_regularizer='l2', name='game_embedding')(game_input)
    
    user_vector = Flatten()(user_embedding)
    game_vector = Flatten()(game_embedding)
    
    concatenated = Concatenate()([user_vector, game_vector])
    
    dense_1 = Dense(512, activation='relu', kernel_regularizer='l2')(concatenated)
    dropout_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(256, activation='relu', kernel_regularizer='l2')(dropout_1)
    dropout_2 = Dropout(0.5)(dense_2)
    dense_3 = Dense(128, activation='relu', kernel_regularizer='l2')(dropout_2)
    dropout_3 = Dropout(0.5)(dense_3)
    dense_4 = Dense(64, activation='relu', kernel_regularizer='l2')(dropout_3)
    dropout_4 = Dropout(0.5)(dense_4)
    output = Dense(1, activation='linear')(dropout_4)
    
    model = Model(inputs=[user_input, game_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    
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

# Calculate RMSE for model evaluation
def calculate_rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def evaluate_knn_model(knn, user_game_matrix_csr, test_data):
    test_data = test_data.copy()
    test_data['user_idx'] = test_data['user_id'].astype('category').cat.codes
    test_data['game_idx'] = test_data['game_title'].astype('category').cat.codes
    
    user_indices = test_data['user_idx'].values
    game_indices = test_data['game_idx'].values
    true_ratings = test_data['rating'].values
    
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
    
    df = load_data(dataset_path)
    
    # KNN Model
    user_game_matrix_csr, user_ids, game_titles = prepare_data_for_knn(df)
    
    train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print("Training base model (KNN)...")
    base_model = train_base_model(user_game_matrix_csr)
    save_model(base_model, 'base_model_knn.pkl')
    
    print("Evaluating base model (KNN)...")
    knn_rmse = evaluate_knn_model(base_model, user_game_matrix_csr, test_data)
    print(f"KNN Model RMSE: {knn_rmse}")
    
    # Neural Network Models
    user_ids, game_ids, ratings = prepare_data_for_nn(df)
    
    user_ids_train, user_ids_temp, game_ids_train, game_ids_temp, ratings_train, ratings_temp = train_test_split(
        user_ids, game_ids, ratings, test_size=0.4, random_state=42
    )
    user_ids_val, user_ids_test, game_ids_val, game_ids_test, ratings_val, ratings_test = train_test_split(
        user_ids_temp, game_ids_temp, ratings_temp, test_size=0.5, random_state=42
    )
    
    # Naive Model
    print("Training naive model (Basic Neural Collaborative Filtering)...")
    num_users = len(np.unique(user_ids))
    num_games = len(np.unique(game_ids))
    
    naive_model = build_naive_model(num_users, num_games)
    naive_model.fit([user_ids_train, game_ids_train], ratings_train, epochs=5, batch_size=64, validation_data=([user_ids_val, game_ids_val], ratings_val))
    save_model(naive_model, 'naive_model.h5')
    
    naive_predictions = naive_model.predict([user_ids_test, game_ids_test]).flatten()
    naive_rmse = calculate_rmse(ratings_test, naive_predictions)
    print(f"Naive Neural Collaborative Filtering Model RMSE: {naive_rmse}")
    
    # Fine-tuned Model
    print("Training fine-tuned model (Enhanced Neural Collaborative Filtering)...")
    fine_tuned_model = build_fine_tuned_model(num_users, num_games)
    
    fine_tuned_model.fit([user_ids_train, game_ids_train], ratings_train, epochs=20, batch_size=64, validation_data=([user_ids_val, game_ids_val], ratings_val))
    save_model(fine_tuned_model, 'fine_tuned_model.h5')
    
    print("Evaluating fine-tuned model (Enhanced Neural Collaborative Filtering)...")
    fine_tuned_rmse = evaluate_nn_model(fine_tuned_model, user_ids_test, game_ids_test, ratings_test)
    print(f"Fine-Tuned Neural Collaborative Filtering Model RMSE: {fine_tuned_rmse}")