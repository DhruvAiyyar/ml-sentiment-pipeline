import pytest
from app import app





@pytest.fixture
def client():
 return app.test_client()


def test_health_check(client):
    response = client.get('/')
    
    assert response.status_code == 200 , f"Expected 200 but got {response.status_code}"

    data=response.json

    assert 'message' in data,"response missing 'message' field"

    print("Health check test passed")


def test_predict_valid_input(client):
    response = client.post(
        '/predict',
        json = {'text': 'I love this movie!'}
    )

    assert response.status_code == 200,f"Expected 200 got {response.status_code}"

    data = response.json

    assert 'sentiment' in data, 'Missing sentiment field'
    assert 'confidence' in data, "missing confidence field"
    assert 'text' in data, "Missing text field"

    confidence = data['confidence']
    assert 0 <= confidence <= 1, f"Confidence {confidence} not in [0,1]"

    print(f"Prediction test passed: {data['sentiment']} ")



def test_predict_empty_text(client):
    response = client.post(
        '/predict',
        json = {'text':''}
    )

    assert response.status_code == 400, f"expected 400 got {response.status_code}"

    data = response.json

    assert 'error' in data
    
    print("Empty text validation works")



def test_predict_missing_field(client):
    response = client.post(
        '/predict',
        json={}
    )
    
    assert response.status_code == 400,f"Expected 400 got {response.status_code}"

    print("Missing field validation works")



def test_predict_negative_sentiment(client):

    response = client.post(
        '/predict',
        json = {'text': 'I hate this terrible product'}
    )

    assert response.status_code == 200
    
    data = response.json

    assert 'sentiment' in data

    print(f"Negative prediction: {data['sentiment']}")







