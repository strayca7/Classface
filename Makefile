PYTHON := uv run python
LFW_URL := https://ndownloader.figshare.com/files/5976015
LFW_ARCHIVE := data/raw/lfw-funneled.tgz

.PHONY: setup download-lfw prepare-dataset preprocess cover clean

## setup: 创建项目所需目录结构
setup:
	mkdir -p data/raw/lfw data/processed/lfw data/overlays \
	          data/features data/results/figures scripts

## download-lfw: 下载 LFW-funneled 数据集并解压至 data/raw/lfw/
download-lfw:
	@echo "下载 LFW-funneled (~200MB)..."
	curl -L -o $(LFW_ARCHIVE) $(LFW_URL)
	tar -xzf $(LFW_ARCHIVE) -C data/raw/
	@mv data/raw/lfw_funneled/* data/raw/lfw/ 2>/dev/null || true
	@rm -rf data/raw/lfw_funneled
	@echo "完成，图像存放于 data/raw/lfw/"

## prepare-dataset: 筛选 ≥2 张图像的身份，生成 gallery/query 分割清单
prepare-dataset:
	$(PYTHON) scripts/prepare_dataset.py

## preprocess: 批量预处理 LFW → data/processed/lfw/
preprocess:
	$(PYTHON) scripts/preprocess.py --src data/raw/lfw --dst data/processed/lfw

## cover: 生成遮挡合成样例图 → data/output/
cover:
	@mkdir -p data/output
	@rm -f data/output/*.jpg
	$(PYTHON) scripts/generate_cover.py $(ARGS)

## clean: 清除所有生成产物
clean:
	rm -rf data/output data/processed data/features data/results data/synthetic
