PYTHON := uv run python
LFW_URL := https://ndownloader.figshare.com/files/5976015
LFW_ARCHIVE := data/raw/lfw-funneled.tgz

.PHONY: setup download-lfw prepare-dataset preprocess \
        segment-skin segment-face eval-seg validate-seg \
        build-gallery eval-baseline \
        gen-overlays generate \
        crop eval-compare \
        cover clean

## setup: 创建项目所需目录结构
setup:
	mkdir -p data/raw/lfw data/processed/lfw data/overlays \
	          data/features data/results/figures scripts \
	          data/segmented/skin_ycrcb data/segmented/skin_gmm \
	          data/segmented/grabcut data/segmented/watershed \
	          data/synthetic/cup data/synthetic/hand data/synthetic/book \
	          data/cropped/gallery data/cropped/query data/cropped/vis

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

## segment-skin: 肤色分割（YCrCb + GMM）→ data/segmented/skin_*/
segment-skin:
	$(PYTHON) scripts/segment_skin.py $(ARGS)

## segment-face: 前景分割（GrabCut + Watershed）→ data/segmented/grabcut/ & watershed/
segment-face:
	$(PYTHON) scripts/segment_face.py $(ARGS)

## eval-seg: 生成分割方法对比图和统计报告
eval-seg:
	$(PYTHON) scripts/eval_segmentation.py $(ARGS)

## validate-seg: 验证第二阶段所有分割输出
validate-seg:
	$(PYTHON) scripts/validate_segmentation.py

## build-gallery: 提取 gallery 身份特征，缓存至 data/features/
build-gallery:
	$(PYTHON) scripts/build_gallery.py $(ARGS)

## eval-baseline: 在干净 LFW query 上评估 Top-1 基线准确率
eval-baseline:
	$(PYTHON) scripts/evaluate.py --mode baseline $(ARGS)

## gen-overlays: 程序化生成 RGBA 遮挡素材 PNG → data/overlays/
gen-overlays:
	$(PYTHON) scripts/gen_overlays.py

## generate: 合成课堂遮挡图像 → data/synthetic/{cup,hand,book}/
generate:
	$(PYTHON) scripts/generate_cover.py $(ARGS)

## crop: 眼周区域裁剪，预计算 gallery_cropped.npy
crop:
	$(PYTHON) scripts/crop.py $(ARGS)

## eval-compare: 三组对比实验（基线 / 遮挡naive / 两级级联）
eval-compare:
	$(PYTHON) scripts/evaluate.py --mode compare $(ARGS)

## cover: 生成遮挡合成样例图 → data/output/
cover:
	@mkdir -p data/output
	@rm -f data/output/*.jpg
	$(PYTHON) scripts/generate_cover.py $(ARGS)

## clean: 清除所有生成产物
clean:
	rm -rf data/output data/processed data/features data/results data/synthetic data/segmented data/cropped
