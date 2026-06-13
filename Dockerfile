FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Auto-index bundled code on startup; restrict indexing to the project dir.
ENV AUTO_INDEX_PATH=backend \
    INDEX_BASE_DIR=/app \
    GEMINI_EMBED_MODEL=gemini-embedding-001 \
    GEMINI_CHAT_MODEL=gemini-2.5-flash

# Most platforms inject $PORT; default to 8000 locally.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
