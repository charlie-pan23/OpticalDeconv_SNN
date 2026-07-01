# Results Directory

This directory separates training artifacts from evaluation artifacts.

## Training artifacts

- `cifar10dvs/`: frozen CIFAR10-DVS checkpoints, configs, and train/test runs.
- `dvsgesture/`: frozen DVS Gesture checkpoints, configs, and train/test runs.

## Evaluation artifacts

- `eval_v2/`: regenerated evaluation outputs for Section 4.
- Each `eval_xx` directory stores raw data only, such as JSON, CSV, YAML, and NPY.
- Paper figures are generated separately under `plot/results/`.

## Archive

- `_archive/`: old evaluation outputs, old frozen snapshots, and deprecated runs.