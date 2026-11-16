FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=9696

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY model.bin .
COPY predict.py .

EXPOSE 9696

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9696/health')" || exit 1

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "--workers=4", "predict:app"]
