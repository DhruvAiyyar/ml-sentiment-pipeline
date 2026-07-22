from flask import Flask, request, jsonify
import pickle
import logging

logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




app = Flask(__name__)  

try:
    with open('models/model.pkl', 'rb') as f:  
        model = pickle.load(f)  
    
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    print("✓ Models loaded successfully")
except FileNotFoundError:
    print("✗ Models not found. Run train_model.py first!")
    exit()




@app.route('/', methods=['GET'])  
def health_check():
    """Health check endpoint"""
    
    logger.info("Health check requested")
    return jsonify({"message": "API is running"})  




@app.route('/predict', methods=['POST'])  
def predict():
    """Make sentiment prediction"""
    
    try:
        data = request.json  
        logger.info("Prediction request received")
        
        
        if 'text' not in data:  
            error_msg = "Missing 'text' field"
            logger.warning(error_msg)
            return jsonify({"error": error_msg}), 400  
        
        text = data.get('text', '')
        
        if not text or len(text.strip()) == 0:
            error_msg = "Text cannot be empty"
            logger.warning(error_msg)
            return jsonify({"error": error_msg}), 400  
        
        
        text_vectorized = vectorizer.transform([text])  
        prediction = model.predict(text_vectorized)[0]  
        confidence = model.predict_proba(text_vectorized).max()  
        
        
        response = {
            "text": text[:100],
            "sentiment": prediction,
            "confidence": float(confidence)
        }
        
        logger.info(f"Prediction: {prediction} (confidence: {confidence:.2f})")
        return jsonify(response), 200  
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500  




if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("STARTING FLASK API")
    print("=" * 50)
    
    app.run(
        host='0.0.0.0',  
        port=8000,  
        debug=True  
    )