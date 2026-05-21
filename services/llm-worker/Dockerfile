FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set the import path to the root of the application
ENV PYTHONPATH="/app"

CMD ["python", "app.py"]