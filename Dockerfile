FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port 10000 explicitly as a fallback
EXPOSE 10000

# Using shell form to correctly resolve Render's dynamic port environment variable if it shifts
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
