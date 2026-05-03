FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and pre-stripped JSON data
COPY . .

# DATA_ROOT: original data root path (used only for constructing HF image URLs from
#            screenshot_path fields in the JSON — not for local file access).
# RUNS_DIR:  where the JSON run data lives inside this image.
# HF_DATASET_REPO: HF Dataset that hosts the screenshot images.
ENV DATA_ROOT=/vepfs-mlp2/project-infoengine/guanhaoxiang/yanxiwang \
    RUNS_DIR=/app/data \
    HF_DATASET_REPO=ShaofantuoshuzhengzhiSha/guiguard-bench-survey-data \
    RATINGS_PATH=/tmp/ratings.jsonl

EXPOSE 7860

CMD ["python3", "server.py", "--host", "0.0.0.0", "--port", "7860"]
