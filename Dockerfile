# 可解释AI实时入侵检测平台 — 后端镜像
# 一个容器承载: FastAPI 后端 + Web 前端面板 + XGBoost 模型
FROM python:3.10-slim

WORKDIR /app

# xgboost 运行依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先拷贝依赖清单并安装(利用层缓存,代码改动无需重装依赖)
COPY detector/requirements.txt /app/detector/requirements.txt
RUN pip install --no-cache-dir -r /app/detector/requirements.txt

# 拷贝检测服务代码
COPY detector /app/detector

# 只打包检测所需模型文件(避免把 30MB+ 的 RF/DL 模型也打进镜像)
RUN mkdir -p /app/models
COPY models/ids_xgb_multi.json       /app/models/ids_xgb_multi.json
COPY models/ids_xgb_multi.meta.json  /app/models/ids_xgb_multi.meta.json

WORKDIR /app/detector

# 环境变量(可被 docker-compose 覆盖)
ENV MODELS_DIR=/app/models \
    DB_PATH=/data/alerts.db

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
