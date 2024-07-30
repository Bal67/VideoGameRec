import os
import pandas as pd
import numpy as np
import pickle
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#Load Dataset
def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
    return df

def prepare_data(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'game_title', 'rating']], reader)
    return data

def prepare_data_for_nn(df):
    user_ids = df['user_id'].astype('category').cat.codes.values
    game_ids = df['game_title'].astype('category').cat.codes.values
    ratings = df['rating'].values
    return user_ids, game_ids, ratings



#Base model created using KNN
def train_base_model(data):
    algo = KNNBasic()
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    return algo


#Naive model created using neural Collaborative Filtering
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


#Fine-tuned Naive Model
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
    model.compile(optimizer=Adam(lr=0.001), loss='mse')
    
    return model



#Save model to desired path
def save_model(model, model_name, model_path='./models'):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    model_file = os.path.join(model_path, model_name)
    
    if isinstance(model, KNNBasic):
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
    else:
        model.save(model_file)
        
    print(f"Model saved to {model_file}")




if __name__ == "__main__":
    dataset_path = './datasets/steam/steam-200k.csv'
    
    # Load the data
    df = load_data(dataset_path)
    
    # Prepare data for base model
    data_prepared = prepare_data(df)
    
    # Train base model
    print("Training base model (KNN)...")
    base_model = train_base_model(data_prepared)
    save_model(base_model, 'base_model_knn.pkl')
    
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
    
    # Train fine-tuned model
    print("Training fine-tuned model (Enhanced Neural Collaborative Filtering)...")
    fine_tuned_model = build_fine_tuned_model(num_users, num_games)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    fine_tuned_model.fit([user_ids_train, game_ids_train], ratings_train, epochs=20, batch_size=64, validation_data=([user_ids_test, game_ids_test], ratings_test), callbacks=[early_stopping])
    save_model(fine_tuned_model, 'fine_tuned_model.h5')
