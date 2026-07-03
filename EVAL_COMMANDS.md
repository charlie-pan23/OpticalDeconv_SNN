# HIPSA Evaluation Commands

This file lists the non-dry-run commands used to reproduce the `5090` branch evaluation pipeline.

Assumptions:

```bash
conda activate snn_photonics
cd /path/to/OpticalDeconv_SNN
```

Windows `cmd` / Anaconda Prompt users can run the same commands on one line. Linux/macOS users can keep the line continuations shown below.

## Common paths

```text
CIFAR10-DVS config:
  configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml

CIFAR10-DVS checkpoint:
  results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth

DVS Gesture config:
  results/dvsgesture/config_dvsgesture_acc88p54.yaml

DVS Gesture checkpoint:
  results/dvsgesture/best_dvsgesture_acc88p54.pth

HIPSA hardware config:
  configs/hardware_hipsa.yaml

Device parameter config:
  configs/device_params.yaml

Evaluation output root:
  results/eval_v2

Final figure output root:
  plot/results/final
```

## eval_00: clean accuracy sanity check

### CIFAR10-DVS

```bash
# Data source:
#   config     = configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml
#   checkpoint = results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth
# Output:
#   results/eval_v2/cifar10dvs/eval_00/
python -m eval.eval_00 \
  --config configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml \
  --checkpoint results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth \
  --output-root results/eval_v2 \
  --split test \
  --batch-size 128 \
  --num-workers 4 \
  --device auto \
  --save-logits
```

### DVS Gesture

```bash
# Data source:
#   config     = results/dvsgesture/config_dvsgesture_acc88p54.yaml
#   checkpoint = results/dvsgesture/best_dvsgesture_acc88p54.pth
# Output:
#   results/eval_v2/dvsgesture/eval_00/
python -m eval.eval_00 \
  --config results/dvsgesture/config_dvsgesture_acc88p54.yaml \
  --checkpoint results/dvsgesture/best_dvsgesture_acc88p54.pth \
  --output-root results/eval_v2 \
  --split test \
  --batch-size 128 \
  --num-workers 4 \
  --device auto \
  --save-logits
```

## eval_01: activity trace and active-SOP statistics

### CIFAR10-DVS

```bash
# Data source:
#   config/checkpoint from frozen CIFAR10-DVS model
# Output:
#   results/eval_v2/cifar10dvs/eval_01/
python -m eval.eval_01 \
  --config configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml \
  --checkpoint results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth \
  --output-root results/eval_v2 \
  --split test \
  --batch-size 128 \
  --num-workers 4 \
  --device auto
```

### DVS Gesture

```bash
# Data source:
#   config/checkpoint from frozen DVS Gesture model
# Output:
#   results/eval_v2/dvsgesture/eval_01/
python -m eval.eval_01 \
  --config results/dvsgesture/config_dvsgesture_acc88p54.yaml \
  --checkpoint results/dvsgesture/best_dvsgesture_acc88p54.pth \
  --output-root results/eval_v2 \
  --split test \
  --batch-size 128 \
  --num-workers 4 \
  --device auto
```

## eval_02: default device-calibrated latency / power / energy

```bash
# Data source:
#   results/eval_v2/cifar10dvs/eval_01/summary.json
#   results/eval_v2/dvsgesture/eval_01/summary.json
#   configs/hardware_hipsa.yaml
#   configs/device_params.yaml
# Output:
#   results/eval_v2/cifar10dvs/eval_02/
#   results/eval_v2/dvsgesture/eval_02/
#   results/eval_v2/combined/eval_02/
# Default design:
#   HAPR group size = 8
#   ADC macros      = 16
python -m eval.eval_02 \
  --datasets cifar10dvs dvsgesture \
  --input-root results/eval_v2 \
  --output-root results/eval_v2 \
  --hardware configs/hardware_hipsa.yaml \
  --device-params configs/device_params.yaml
```

## eval_03: comparator threshold sensitivity

### CIFAR10-DVS

```bash
# Data source:
#   config/checkpoint from frozen CIFAR10-DVS model
#   configs/hardware_hipsa.yaml
#   configs/device_params.yaml
# Output:
#   results/eval_v2/cifar10dvs/eval_03/
python -m eval.eval_03 \
  --dataset cifar10dvs \
  --config configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml \
  --checkpoint results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth \
  --output-root results/eval_v2 \
  --thresholds 0.0 0.01 0.02 0.05 0.10 0.20 \
  --batch-size 128 \
  --num-workers 4 \
  --device auto
```

### DVS Gesture

```bash
# Data source:
#   config/checkpoint from frozen DVS Gesture model
#   configs/hardware_hipsa.yaml
#   configs/device_params.yaml
# Output:
#   results/eval_v2/dvsgesture/eval_03/
python -m eval.eval_03 \
  --dataset dvsgesture \
  --config results/dvsgesture/config_dvsgesture_acc88p54.yaml \
  --checkpoint results/dvsgesture/best_dvsgesture_acc88p54.pth \
  --output-root results/eval_v2 \
  --thresholds 0.0 0.01 0.02 0.05 0.10 0.20 \
  --batch-size 128 \
  --num-workers 4 \
  --device auto
```

## eval_04: HAPR / ADC pool / MRR sensitivity

```bash
# Data source:
#   results/eval_v2/cifar10dvs/eval_01/summary.json
#   results/eval_v2/dvsgesture/eval_01/summary.json
#   configs/hardware_hipsa.yaml
#   configs/device_params.yaml
# Output:
#   results/eval_v2/cifar10dvs/eval_04/
#   results/eval_v2/dvsgesture/eval_04/
#   results/eval_v2/combined/eval_04/
# Sweeps:
#   ADC pool sizes   = 8, 16, 32, 64, 128
#   HAPR group sizes = 4, 8, 16, 32
python -m eval.eval_04 \
  --datasets cifar10dvs dvsgesture \
  --input-root results/eval_v2 \
  --output-root results/eval_v2 \
  --hardware configs/hardware_hipsa.yaml \
  --device-params configs/device_params.yaml \
  --adc-pool-sizes 8 16 32 64 128 \
  --hapr-group-sizes 4 8 16 32
```

### Optional all-biased ADC upper-bound rerun

```bash
# Data source:
#   same as eval_04
# Output:
#   results/eval_v2_allbiased/<dataset>/eval_04/
python -m eval.eval_04 \
  --datasets cifar10dvs dvsgesture \
  --input-root results/eval_v2 \
  --output-root results/eval_v2_allbiased \
  --hardware configs/hardware_hipsa.yaml \
  --device-params configs/device_params.yaml \
  --adc-pool-sizes 8 16 32 64 128 \
  --hapr-group-sizes 4 8 16 32 \
  --adc-power-mode all_biased
```

## eval_05: device-specific robustness

### CIFAR10-DVS

```bash
# Data source:
#   config/checkpoint from frozen CIFAR10-DVS model
#   configs/hardware_hipsa.yaml
#   configs/device_params.yaml
# Output:
#   results/eval_v2/cifar10dvs/eval_05/
# Mode/seeds:
#   full sweep, seeds 0 1 2
# Design context:
#   HAPR group size = 16
#   ADC macros      = 32
python -m eval.eval_05 \
  --dataset cifar10dvs \
  --config configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml \
  --checkpoint results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth \
  --output-root results/eval_v2 \
  --mode full \
  --batch-size 16 \
  --num-workers 0 \
  --device auto \
  --hapr-group-size 16 \
  --adc-macros 32 \
  --seeds 0 1 2
```

### DVS Gesture

```bash
# Data source:
#   config/checkpoint from frozen DVS Gesture model
#   configs/hardware_hipsa.yaml
#   configs/device_params.yaml
# Output:
#   results/eval_v2/dvsgesture/eval_05/
# Mode/seeds:
#   full sweep, seeds 0 1 2
# Design context:
#   HAPR group size = 16
#   ADC macros      = 32
python -m eval.eval_05 \
  --dataset dvsgesture \
  --config results/dvsgesture/config_dvsgesture_acc88p54.yaml \
  --checkpoint results/dvsgesture/best_dvsgesture_acc88p54.pth \
  --output-root results/eval_v2 \
  --mode full \
  --batch-size 32 \
  --num-workers 0 \
  --device auto \
  --hapr-group-size 16 \
  --adc-macros 32 \
  --seeds 0 1 2
```

## eval_06: local CPU/GPU runtime and energy baseline

Run these on the machine whose CPU/GPU baseline will be reported.

### CIFAR10-DVS local baseline

```bash
# Data source:
#   config/checkpoint from frozen CIFAR10-DVS model
# Output:
#   results/eval_v2/cifar10dvs/eval_06/
#   plot/results/eval_06_local/ after plotting
# Active power:
#   CPU = 73.55 W
#   GPU = 69.99 W
python -m eval.eval_06 \
  --dataset cifar10dvs \
  --config configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml \
  --checkpoint results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth \
  --output-root results/eval_v2 \
  --devices cpu cuda \
  --batch-size 1 \
  --num-workers 0 \
  --num-warmup 20 \
  --num-runs 100 \
  --cpu-active-power-w 73.55 \
  --gpu-active-power-w 69.99
```

### DVS Gesture local baseline

```bash
# Data source:
#   config/checkpoint from frozen DVS Gesture model
# Output:
#   results/eval_v2/dvsgesture/eval_06/
#   plot/results/eval_06_local/ after plotting
# Active power:
#   CPU = 72.17 W
#   GPU = 57.95 W
python -m eval.eval_06 \
  --dataset dvsgesture \
  --config results/dvsgesture/config_dvsgesture_acc88p54.yaml \
  --checkpoint results/dvsgesture/best_dvsgesture_acc88p54.pth \
  --output-root results/eval_v2 \
  --devices cpu cuda \
  --batch-size 1 \
  --num-workers 0 \
  --num-warmup 20 \
  --num-runs 100 \
  --cpu-active-power-w 72.17 \
  --gpu-active-power-w 57.95
```

## Stage plot commands

### plot_00: confusion matrices

```bash
# Data source:
#   results/eval_v2/cifar10dvs/eval_00/confusion_matrix.csv
# Output:
#   plot/results/eval_00/cifar10dvs/
python -m plot.plot_00 \
  --input results/eval_v2/cifar10dvs/eval_00 \
  --dataset cifar10dvs

# Data source:
#   results/eval_v2/dvsgesture/eval_00/confusion_matrix.csv
# Output:
#   plot/results/eval_00/dvsgesture/
python -m plot.plot_00 \
  --input results/eval_v2/dvsgesture/eval_00 \
  --dataset dvsgesture
```

### plot_01: workload activity

```bash
# Data source:
#   results/eval_v2/<dataset>/eval_01/summary.json
# Output:
#   plot/results/eval_01/
python -m plot.plot_01 \
  --input-root results/eval_v2 \
  --output-root plot/results/eval_01 \
  --datasets cifar10dvs dvsgesture
```

### plot_02 and plot_03: eval_02 latency/energy and power breakdown

```bash
# Data source:
#   results/eval_v2/<dataset>/eval_02/latency_energy_summary.json
# Output:
#   plot/results/eval_02/
python -m plot.plot_02 \
  --input-root results/eval_v2 \
  --output-root plot/results/eval_02 \
  --datasets cifar10dvs dvsgesture

# Data source:
#   results/eval_v2/<dataset>/eval_02/power_breakdown.csv
# Output:
#   plot/results/eval_02/
python -m plot.plot_03 \
  --input-root results/eval_v2 \
  --output-root plot/results/eval_02 \
  --datasets cifar10dvs dvsgesture
```

### plot_04: comparator threshold sweep

```bash
# Data source:
#   results/eval_v2/<dataset>/eval_03/threshold_sweep.csv
# Output:
#   plot/results/eval_03/
python -m plot.plot_04 \
  --input-root results/eval_v2 \
  --output-root plot/results/eval_03 \
  --datasets cifar10dvs dvsgesture
```

### plot_05: HAPR / ADC / MRR sensitivity

```bash
# Data source:
#   results/eval_v2/<dataset>/eval_04/adc_pool_sweep.csv
#   results/eval_v2/<dataset>/eval_04/hapr_adc_sweep.csv
#   results/eval_v2/<dataset>/eval_04/mrr_sensitivity.csv
# Output:
#   plot/results/eval_04/
python -m plot.plot_05 \
  --input-root results/eval_v2 \
  --output-root plot/results/eval_04 \
  --datasets cifar10dvs dvsgesture
```

### plot_06: CPU/GPU/HIPSA local comparison

```bash
# Data source:
#   results/eval_v2/<dataset>/eval_06/runtime_summary.json
#   results/eval_v2/<dataset>/eval_04/hapr_adc_sweep.csv
# Output:
#   plot/results/eval_06_local/
python -m plot.plot_06 \
  --input-root results/eval_v2 \
  --output-root plot/results/eval_06_local \
  --datasets cifar10dvs dvsgesture \
  --hipsa-hapr 16 \
  --hipsa-adc 32
```

### plot_07: robustness sweep / summary

```bash
# Data source:
#   results/eval_v2/<dataset>/eval_05/robustness_summary.csv
# Output:
#   plot/results/eval_05_1/
python -m plot.plot_07_final \
  --input-root results/eval_v2 \
  --output-root plot/results/eval_05_1 \
  --datasets cifar10dvs dvsgesture \
  --view practical \
  --errorbar none
```

## Final paper figure commands

### Final Fig. 5

```bash
# Data source:
#   results/eval_v2/<dataset>/eval_01/summary.json
#   plot/results/eval_06_local/plot_06_comparison_data.csv
# Output:
#   plot/results/final/fig5_activity_performance.pdf
#   plot/results/final/fig5a_activity.pdf
#   plot/results/final/fig5b_latency.pdf
#   plot/results/final/fig5c_energy.pdf
python -m plot.plot_final_fig5
```

### Final Fig. 6

```bash
# Data source:
#   results/eval_v2/<dataset>/eval_01/summary.json
#   results/eval_v2/<dataset>/eval_04/hapr_adc_sweep.csv
#   configs/hardware_hipsa.yaml
#   configs/device_params.yaml
# Output:
#   plot/results/final/fig6_power_adc_hapr.pdf
#   plot/results/final/fig6a_power_breakdown.pdf
#   plot/results/final/fig6b_hapr_adc_energy.pdf
#   plot/results/final/fig6c_adc_utilization.pdf
python -m plot.plot_final_fig6
```

### Final Fig. 7

```bash
# Data source:
#   plot/results/eval_05_1/plot_07_aggregated_data_practical_none_hybrid.csv
# Output:
#   plot/results/final/fig7_robustness_summary.pdf
#   plot/results/final/fig7_robustness_summary_panel.pdf
python -m plot.plot_final_fig7
```

### Final figures, one command

```bash
# Data source:
#   all required eval_v2 outputs and final plot CSVs
# Output:
#   plot/results/final/
python -m plot.plot_final_fig5
python -m plot.plot_final_fig6
python -m plot.plot_final_fig7
```

## Eval completeness check

```bash
# Data source:
#   results/eval_v2/
# Output:
#   terminal status table
python scripts/check_eval_v2.py
```

## LaTeX integration

Final plot scripts generate both combined preview figures and separate title-free panels:

```text
fig5_activity_performance.pdf
fig5a_activity.pdf
fig5b_latency.pdf
fig5c_energy.pdf

fig6_power_adc_hapr.pdf
fig6a_power_breakdown.pdf
fig6b_hapr_adc_energy.pdf
fig6c_adc_utilization.pdf

fig7_robustness_summary.pdf
fig7_robustness_summary_panel.pdf
```

Use the separate panel PDFs with LaTeX `subfigure` / `subcaption` so that subfigure titles and captions use the paper template's font.
