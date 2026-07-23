from monitoring import PredictionMonitor
monitor = PredictionMonitor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        if 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
        confidence = model.predict_proba(text_vec).max()
        
        # LOG THE PREDICTION
        monitor.log_prediction(text, prediction, float(confidence))
        
        logger.info(f"Prediction: {prediction}")
        return jsonify({
            "text": text,
            "sentiment": prediction,
            "confidence": float(confidence)
        })
    
    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500
    