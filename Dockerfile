FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train_model.py .
COPY app.py .
COPY monitoring.py .
COPY data/ data/
COPY logs/ logs/

RUN python train_model.py

EXPOSE 8000

CMD ["python", "app.py"]
