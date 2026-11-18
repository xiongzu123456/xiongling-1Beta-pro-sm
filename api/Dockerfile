FROM python:3.12-slim

WORKDIR /app
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# 镜像里没有已训练模型时，构建阶段直接练一个，保证服务启动即有模型
RUN python3 train.py --backend numpy --data data/sample.txt --out model --epochs 3 || true

ENV XL_API_KEY=sk-xionglingautobaiduscshumi
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT}"]
