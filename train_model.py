"""Training script for ML sentiment model"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import logging
import os


import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # Remove URLs
    text = re.sub(r"@\w+", "", text)         # Remove @mentions
    text = re.sub(r"[^a-z\s]", " ", text)    # Keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data():
    file_path = os.path.join(BASE_DIR, "data", "twitter_training.csv")
    df = pd.read_csv(file_path,header=None)
    df.columns = ["id", "entity", "sentiment", "text"]
    df.dropna(inplace=True)
    df["text"] = df["text"].apply(clean_text)
    logger.info(f"Loaded {len(df)} sample")

    return df


def explore_data(df):
    print("="*50)
    print("DATA EXPLORATION")
    print("="*50)

    print(f"Total rows and coloumns {df.shape}")
    print(f"Total coloumns {df.columns}")
    print(df.head())
    print(f"Unique values {df['sentiment'].unique()}")
    print(f"Missing values {df.isnull().sum()}")



def prepare_data(df):
    X = df['text']
    y = df['sentiment']

    logger.info("data prepared")
    return X,y


def vectorize_text(X_train,X_test):
    vectorizer = TfidfVectorizer(max_features=10000,stop_words="english",ngram_range=(1,2))

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    logger.info(f"Text vectorized. Shape : {X_train_vec.shape}")
    return X_train_vec,X_test_vec,vectorizer


def split_data(X,y):
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,
        test_size = 0.2,
        random_state=42

    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"test set : {len(X_test)}")
    return X_train , X_test , y_train , y_test


def train_model(X_train , y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train , y_train)
    logger.info("Model trained")
    return model


def evaluate_model(model , X_test , y_test):
    accuracy = model.score(X_test,y_test)
    print(f"\n{'='*50}")
    print(f"MODEL ACCURACY: {accuracy:.2%}")
    print(f"{'='*50}")

    if accuracy >= 0.8:
        print("good accuracy")
        logger.info(f"Model accuracy")
    else:
        print("Accuracy below 80%.Need more data or better preprocessing")
        logger.warning(f"Model accuracy: {accuracy:.2%}")

    return accuracy


def save_model(model,vectorizer):
    with open('models/model.pkl','wb') as f:
        pickle.dump(model,f)

    with open('models/vectorizer.pkl','wb') as f:
        pickle.dump(vectorizer,f)

    print("model saved")
    print("vectorizer saved ")
    logger.info("Model and vectorizer saved")


def test_model_loading():
    with open ('models/model.pkl','rb') as f:
        loaded_model = pickle.load(f)
    
    with open('models/vectorizer.pkl' , 'rb') as f:
        loaded_vectorizer = pickle.load(f)

    print("model loaded")
    logger.info("model loading test passed")
    return loaded_model,loaded_vectorizer


if __name__ == "__main__":
    print("\n" + "=" *50)
    print("Training ML model")
    print("\n" + "=" *50)


    df = load_data()
    explore_data(df)
    X,y = prepare_data(df)
    X_train , X_test , y_train ,y_test = split_data(X,y)
    X_train_vec , X_test_vec , vectorizer = vectorize_text(X_train,X_test)
    model = train_model(X_train_vec,y_train)
    accuracy = evaluate_model(model,X_test_vec,y_test)
    save_model(model,vectorizer)
    test_model_loading()
    print("training complete")








    




def main():
    print("=" * 50)
    print("TRAINING ML MODEL")
    print("=" * 50)
    logger.info("Training started")
    # TODO: Fill in training code
    print("✓ Training complete!")






