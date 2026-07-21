"""Flask API for sentiment prediction"""

from flask import Flask
import logging

logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def hello():
    return {"message": "Flask API"}


if __name__ == '__main__':
    print("STARTING FLASK API")
    # TODO: Fill in API code
