# 🚀 ML Sentiment Analysis Pipeline

An end-to-end Machine Learning Sentiment Analysis Pipeline built using Python and Flask that classifies Twitter text into sentiment categories. This project follows software engineering best practices by incorporating Docker, automated testing, logging, and CI/CD with GitHub Actions.

---

## 📌 Project Overview

This project demonstrates how a Machine Learning model can be transformed from a simple notebook experiment into a production-ready application.

The pipeline performs:

- Data preprocessing
- Feature extraction using TF-IDF
- Model training
- Model serialization
- REST API inference
- Docker containerization
- Automated testing
- Continuous Integration using GitHub Actions

---

## ✨ Features

- Clean data preprocessing pipeline
- TF-IDF feature engineering
- Random Forest classifier
- REST API using Flask
- Model persistence using Pickle
- Structured logging
- Docker support
- Docker Compose support
- Automated unit testing with PyTest
- GitHub Actions CI pipeline
- Production-ready project structure

---

## 📂 Project Structure

```
ml-sentiment-pipeline/
│
├── app.py
├── train.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── dataset/
│   └── twitter_training.csv
│
├── models/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── tests/
│   └── test_api.py
│
├── logs/
│   └── app.log
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
└── README.md
```

---

## 🛠 Tech Stack

### Programming

- Python

### Machine Learning

- Scikit-learn
- TF-IDF Vectorizer
- Random Forest Classifier

### Backend

- Flask

### DevOps

- Docker
- Docker Compose
- GitHub Actions

### Testing

- PyTest

### Utilities

- Pandas
- NumPy
- Pickle
- Logging

---

## 📊 Dataset

Dataset used:

**Twitter Training Dataset**

The dataset contains labelled tweets classified into different sentiment categories.

Sentiment classes include:

- Positive
- Negative

Dataset Source:

https://www.kaggle.com/datasets

---

## ⚙️ Machine Learning Pipeline

```
Twitter Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Text Preprocessing
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Model Training
(Random Forest Classifier)
        │
        ▼
Model Evaluation
        │
        ▼
Save Model
(.pkl)
        │
        ▼
Flask REST API
        │
        ▼
Docker Container
        │
        ▼
GitHub Actions CI
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/ml-sentiment-pipeline.git

cd ml-sentiment-pipeline
```

Create a virtual environment

```bash
python -m venv venv
```

Activate environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train_model.py
```

The trained model and TF-IDF vectorizer will be stored inside the `models/` directory.

---

## ▶️ Run the Flask API

```bash
python app.py
```

Default server

```
http://localhost:8000
```

---

## 🐳 Run with Docker

Build the image

```bash
docker build -t sentiment-api .
```

Run container

```bash
docker run -p 8000:8000 sentiment-api
```

---

## 🐳 Docker Compose

Start services

```bash
docker compose up --build
```

Stop services

```bash
docker compose down
```

---

## 📡 API Endpoint

### POST /predict

Predict sentiment from input text.

Request

```json
{
    "text": "This product is amazing!"
}
```

Example Response

```json
{
    "prediction": "Positive",
    "confidence": 0.94
}
```

---

## 🧪 Running Tests

Run all tests

```bash
pytest
```

---

## 🔄 Continuous Integration

The project includes GitHub Actions for automated CI.

The workflow automatically:

- Installs dependencies
- Runs unit tests
- Verifies application build
- Ensures code quality before merge

---

## 📈 Model Performance

| Metric | Value |
|---------|--------|
| Model | Random Forest Classifier |
| Feature Extraction | TF-IDF |
| Accuracy | 73% |

---

## 📜 Logging

Application logs are automatically stored in

```
logs/app.log
```

Logs include:

- API requests
- Predictions
- Errors
- Exceptions

---

## 🎯 Future Improvements

- Deploy on AWS/GCP/Azure
- Add Swagger API documentation
- Implement model monitoring
- Add Prometheus & Grafana
- Experiment with transformer-based models (e.g., BERT)
- Add authentication for API endpoints

---

## 🤝 Contributing

Contributions are welcome.

Feel free to fork this repository, create a feature branch, and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Dhruv Aiyyar**

Computer Science Engineering Student

Interested in:

- Machine Learning
- AI Engineering
- MLOps
- Computer Vision
- Software Development

---

⭐ If you found this project useful, consider giving it a star!
