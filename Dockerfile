FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY version.txt .
COPY templates/ templates/

EXPOSE 2463

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:2463/health', timeout=4).status == 200 else 1)"

CMD ["gunicorn", "--workers", "2", "--timeout", "600", "--bind", "0.0.0.0:2463", "app:app"]
