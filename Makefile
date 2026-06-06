PYTHON := uv run python
LFW_URL := https://ndownloader.figshare.com/files/5976015
LFW_ARCHIVE := data/raw/lfw-funneled.tgz

.PHONY: sync setup download-lfw prepare-dataset preprocess \
        segment-skin segment-face eval-seg validate-seg \
        dl-train dl-segment dl-eval-seg \
        build-gallery eval-baseline \
        gen-overlays generate \
        crop eval-compare \
        cover clean \
        plot-results demo-report update-reports

sync:
	uv sync

## setup: 创建项目所需目录结构
setup:
	mkdir -p data/raw/lfw data/processed/lfw data/overlays \
	          data/features data/results/figures scripts \
	          data/segmented/skin_ycrcb data/segmented/skin_gmm \
	          data/segmented/grabcut data/segmented/watershed data/segmented/dl_unet \
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

## dl-train: 训练 ResUNet 人脸分割模型（GrabCut 伪标签，默认 20 epoch）
dl-train:
	$(PYTHON) scripts/dl_train.py $(ARGS)

## dl-segment: 批量推理，输出掩膜至 data/segmented/dl_unet/
dl-segment:
	$(PYTHON) scripts/dl_segment.py $(ARGS)

## dl-eval-seg: DL vs 传统方法对比评估（IoU / Dice / 前景占比）
dl-eval-seg:
	$(PYTHON) scripts/dl_eval_segmentation.py $(ARGS)

## build-gallery: 提取 gallery 身份特征，缓存至 data/features/
build-gallery:
	$(PYTHON) scripts/build_gallery.py $(ARGS)

## eval-baseline: 在干净 LFW query 上评估 Top-1 基线准确率
eval-baseline:
	$(PYTHON) scripts/evaluate.py --mode baseline $(ARGS)

## gen-overlays: （已废弃）overlays 现由 data/overlays/ 中的真实图片资产提供，无需生成
gen-overlays:
	@echo "overlays are now managed as git assets in data/overlays/ — nothing to generate."

## generate: 合成遮挡图像 → data/synthetic/{cup,glasses,sunglasses}/
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

## plot-results: 从实验结果生成图表 → docs/figures/
plot-results:
	$(PYTHON) scripts/plot_results.py $(ARGS)

## demo-report: 生成演示报告图表 + docs/demo_report.md
demo-report:
	$(PYTHON) scripts/generate_demo_report.py $(ARGS)

## update-reports: 解析实验结果并更新所有报告文档与图表
update-reports:
	$(PYTHON) scripts/update_reports.py $(ARGS)

## clean: 清除所有生成产物
clean:
	rm -rf data/output data/processed data/features data/results data/synthetic data/segmented data/cropped
