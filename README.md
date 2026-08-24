# object-detection-benchmarking-and-analysis

Standardized benchmarking framework for SSD300, Faster R-CNN, and YOLOv5 on Pascal VOC 2012.

## Dataset setup (Pascal VOC 2012)

1. Download dataset:
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
2. Extract dataset:
```bash
mkdir -p data/
tar -xvf VOCtrainval_11-May-2012.tar -C data/
```
3. Expected structure:
```
data/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        ├── Annotations/
        └── ImageSets/
```

## Standard benchmark architecture

Pipeline:

`Dataset → Config → Model Adapter → Inference Runner → Standardized Evaluator → Timing Profiler → Visualization/Artifacts → Comparison Aggregator → Report outputs`

Framework code is under `/benchmark`, and any notebooks should be thin wrappers only.

## Canonical config and runs

Main config files:
- `/home/runner/work/object-detection-benchmarking-and-analysis/object-detection-benchmarking-and-analysis/benchmark/configs/default.yaml`
- `/home/runner/work/object-detection-benchmarking-and-analysis/object-detection-benchmarking-and-analysis/benchmark/configs/paper_main.yaml`

Run full benchmark:
```bash
python -m benchmark.run_benchmark --config benchmark/configs/default.yaml
```

Run standardized 20-image comparison:
```bash
python -m benchmark.run_comparison20 --config benchmark/configs/default.yaml
```

Aggregate one run directory:
```bash
python -m benchmark.aggregate_results --results-root results/<run_id>
```

## Standardization guarantees

- Shared adapter interface for all models.
- Shared metrics implementation (`mAP@0.5`, `mAP@0.5:0.95`, precision, recall).
- Shared timing protocol (`preprocessing_ms`, `inference_ms`, `postprocessing_ms`, `total_ms`).
- Shared 20-image protocol via `comparison_images.json`.
- Shared output schema and run metadata capture.

## Notes

- `comparison_images.json` is the canonical image list and order for 20-image comparison runs.
- Results are written to `results/<run_id>/<model>/metrics.json` plus summary artifacts.
- Dataset files and tar archives should remain excluded from Git.
