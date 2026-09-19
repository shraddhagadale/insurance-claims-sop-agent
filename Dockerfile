FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/insurance_claims
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn insurance_claims.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
