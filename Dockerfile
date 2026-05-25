FROM python:3.12-slim

WORKDIR /app

ENV MILVUS_URL=milvus-standalone:19530
ENV REDIS_URL=redis://redis:6379/0
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY src/ ./src/
COPY prompts/ ./prompts/

EXPOSE 8347

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8347/docs || exit 1

CMD ["uvicorn","src.app:app","--host","0.0.0.0","--port","8347"]

