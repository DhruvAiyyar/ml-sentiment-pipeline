import pytest
import pickle
from pathlib import Path


def test_model_loads():
    assert Path('models/model.pkl').exists()
    try:
        with open('models/model.pkl','rb') as f:
            model = pickle.load(f)

        assert model is not None
        print("Model loaded successfully")

    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")



def test_model_accuracy():
    with open('models/model.pkl','rb') as f:
        model = pickle.load(f)

    with open('models/vectorizer.pkl','rb') as f:
        vectorizer = pickle.load(f)

    import pandas as pd
    df = pd.read_csv(
    'data/twitter_training.csv',
    header=None,
    names=['id', 'entity', 'sentiment', 'text']
    )

    df = df.dropna(subset=['text'])

    from sklearn.model_selection import train_test_split
    X = df['text']
    y = df['sentiment']
    X,X_test,y,y_test = train_test_split(
        X,y,test_size=0.2,
        random_state=42
    )

    X_test_vec = vectorizer.transform(X_test)

    accuracy = model.score(X_test_vec,y_test)

    print(f"Model accuracy: {accuracy:.2%}")

    assert accuracy >= 0.8,f"Accuracy {accuracy:.2%} is below 80%"


    def test_vectorizer():
        with open('models/vectorizer.pkl') as f:
            vectorizer = pickle.load(f)

        sample_text = ["I love this"]

        result = vectorizer.transform(sample_text)

        assert result is not None, "Vectorizer returned None"

        assert result.shape[1]== 1000,"Wrong number of features"

        print("vectorizer works correctly")


    def test_prediction_format():
        with open('models/model.pkl','rb') as f:
            model = pickle.load(f)

        with open('models/vectorizer.pkl','rb') as f:
            vectorizer = pickle.load(f)

        sample = vectorizer.transform(["Good movie"])

        prediction = model.predict(sample)[0]

        assert isinstance(prediction , str), "Prediction  is not string"

        print(f"Prediction format valid:{prediction}")




