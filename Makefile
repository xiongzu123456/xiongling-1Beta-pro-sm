.PHONY: help setup train serve test package clean
KEY ?= sk-xionglingautobaiduscshumi
PORT ?= 8000

help:
	@echo "make setup    安装依赖"
	@echo "make train    训练模型（numpy 冒烟后端）"
	@echo "make serve    启动 API  http://127.0.0.1:$(PORT)"
	@echo "make test     单元测试 + 接口测试"
	@echo "make package  打包成 zip"

setup:
	pip install -r api/requirements.txt pytest

train:
	python3 train.py --backend numpy --data data/sample.txt --out model --epochs 3

serve:
	python3 -m uvicorn api.server:app --host 0.0.0.0 --port $(PORT)

test:
	python3 -m pytest tests/ -q
	./scripts/test_api.sh http://127.0.0.1:$(PORT) $(KEY)

package:
	cd .. && zip -qr xiongling-1Beta-pro-sm.zip xiongling-1Beta-pro-sm \
	  -x "*/__pycache__/*" "*/.git/*" "*.pyc"

clean:
	rm -rf generated/*.md __pycache__ api/__pycache__ tests/__pycache__
